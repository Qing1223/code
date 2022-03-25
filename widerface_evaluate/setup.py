"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('bbox', sources=['box_overlaps.pyx'], include_dirs=[numpy.get_include()])
print('=========================')
setup(ext_modules=cythonize([package]))

#
# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# ext_modules = [Extension("bbox", ["box_overlaps.pyx"])]
# setup(
#   name = 'box_overlaps',
#   cmdclass = {'build_ext': build_ext},
#   ext_modules = ext_modules
# )
