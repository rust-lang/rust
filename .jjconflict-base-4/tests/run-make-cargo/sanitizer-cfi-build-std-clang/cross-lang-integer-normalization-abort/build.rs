include!("../shared_build_rs.rs");

fn main() {
    build_foo_static_lib(&["-fsanitize-cfi-icall-experimental-normalize-integers"]);
}
