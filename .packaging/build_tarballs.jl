using BinaryBuilder, Pkg

name = "Enzyme"
repo = "https://github.com/wsmoses/Enzyme.git"

auto_version = "%ENZYME_VERSION%"
version = VersionNumber(split(auto_version, "/")[end])

# Collection of sources required to build attr
sources = [GitSource(repo, "%ENZYME_HASH%")]

# These are the platforms we will build for by default, unless further
# platforms are passed in on the command line
platforms = expand_cxxstring_abis(supported_platforms())

# Bash recipe for building across all platforms
script = raw"""
cd Enzyme
# install_license LICENSE.TXT
CMAKE_FLAGS=()
# Release build for best performance
CMAKE_FLAGS+=(-DENZYME_EXTERNAL_SHARED_LIB=ON)
CMAKE_FLAGS+=(-DCMAKE_BUILD_TYPE=RelWithDebInfo)
# Install things into $prefix
CMAKE_FLAGS+=(-DCMAKE_INSTALL_PREFIX=${prefix})
# Explicitly use our cmake toolchain file and tell CMake we're cross-compiling
CMAKE_FLAGS+=(-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN})
CMAKE_FLAGS+=(-DCMAKE_CROSSCOMPILING:BOOL=ON)
# Tell CMake where LLVM is
CMAKE_FLAGS+=(-DLLVM_DIR="${prefix}/lib/cmake/llvm")
# Force linking against shared lib
CMAKE_FLAGS+=(-DLLVM_LINK_LLVM_DYLIB=ON)
# Build the library
CMAKE_FLAGS+=(-DBUILD_SHARED_LIBS=ON)
cmake -B build -S enzyme -GNinja ${CMAKE_FLAGS[@]}
ninja -C build -j ${nproc} install
"""

# The products that we will ensure are always built
products = Product[
    LibraryProduct(["libEnzyme-9", "libEnzyme"], :libEnzyme),
]

dependencies = [
    BuildDependency(PackageSpec(name="LLVM_full_jll", version=v"11.0.1")),
#    Dependency(PackageSpec(name="libLLVM_jll", version=v"9.0.1"))
]


# Build the tarballs.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               preferred_gcc_version=v"8", julia_compat="1.6")
