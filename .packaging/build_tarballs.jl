using BinaryBuilder, Pkg
using Base.BinaryPlatforms

include("../../fancy_toys.jl")

name = "Enzyme"
repo = "https://github.com/wsmoses/Enzyme.git"

auto_version = "%ENZYME_VERSION%"
version = VersionNumber(split(auto_version, "/")[end])

# Collection of sources required to build attr
sources = [GitSource(repo, "%ENZYME_HASH%")]

# Bash recipe for building across all platforms
script = raw"""
cd Enzyme
# install_license LICENSE.TXT
CMAKE_FLAGS=()
CMAKE_FLAGS+=(-DENZYME_EXTERNAL_SHARED_LIB=ON)
CMAKE_FLAGS+=(-DENZYME_CLANG=OFF)
# RelWithDebInfo for decent performance, with debugability
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
if [[ "${target}" == x86_64-apple* ]]; then
  CMAKE_FLAGS+=(-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.12)
fi
cmake -B build -S enzyme -GNinja ${CMAKE_FLAGS[@]}
ninja -C build -j ${nproc} install
"""

function configure(julia_version, llvm_version)
    # These are the platforms we will build for by default, unless further
    # platforms are passed in on the command line
    platforms = expand_cxxstring_abis(supported_platforms(; experimental=julia_version>=v"1.7"))

    foreach(platforms) do p
        BinaryPlatforms.add_tag!(p.tags, "julia_version", string(julia_version))
    end

    # The products that we will ensure are always built
    products = Product[
        LibraryProduct(["libEnzyme-$(llvm_version.major)", "libEnzyme"], :libEnzyme),
    ]


    dependencies = [
        BuildDependency(get_addable_spec("LLVM_full_jll", llvm_version))
    #    Dependency(PackageSpec(name="libLLVM_jll", version=v"9.0.1")) is given through julia_version tag
    ]

    return platforms, products, dependencies
end

# TODO: Don't require build-id on LLVM version
supported = (
    (v"1.6", v"11.0.1+3"),
    (v"1.7", v"12.0.0+0"),
    (v"1.8", v"13.0.0+3"),
)


for (julia_version, llvm_version) in supported
    platforms, products, dependencies = configure(julia_version, llvm_version)

    any(should_build_platform.(triplet.(platforms))) || continue

    # Build the tarballs.
    build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
                preferred_gcc_version=v"8", julia_compat="1.6")
end
