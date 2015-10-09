extern crate gcc;

fn main() {
    let sources = &[
        "rust_builtin.c",
        "rust_android_dummy.c"
    ];

    gcc::compile_library("librust_builtin.a", sources);
}
