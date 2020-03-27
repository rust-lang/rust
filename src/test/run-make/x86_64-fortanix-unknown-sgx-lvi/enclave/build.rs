fn main() {
    cc::Build::new()
        .file("foo.c")
        .compile("foo_c");

    cc::Build::new()
        .cpp(true)
        .cpp_set_stdlib(None)
        .file("foo_cxx.cpp")
        .compile("foo_cxx");
}
