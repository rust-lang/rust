fn main() {
    cc::Build::new().file("foo.c").compile("foo_c");

    cc::Build::new().file("foo_asm.s").compile("foo_asm");

    cc::Build::new().cpp(true).cpp_set_stdlib(None).file("foo_cxx.cpp").compile("foo_cxx");

    // When the cmake crate detects the clang compiler, it passes the
    //  "--target" argument to the linker which subsequently fails. The
    //  `CMAKE_C_COMPILER_FORCED` option makes sure that `cmake` does not
    //  tries to test the compiler. From version 3.6 the option
    //  `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` can be used
    //  https://cmake.org/cmake/help/v3.5/module/CMakeForceCompiler.html
    let dst = cmake::Config::new("libcmake_foo")
        .build_target("cmake_foo")
        .define("CMAKE_C_COMPILER_FORCED", "1")
        .define("CMAKE_CXX_COMPILER_FORCED", "1")
        .define("CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY", "1")
        .build();
    println!("cargo:rustc-link-search=native={}/build/", dst.display());
    println!("cargo:rustc-link-lib=static=cmake_foo");
}
