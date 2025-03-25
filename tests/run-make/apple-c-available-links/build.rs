fn main() {
    println!("cargo:rerun-if-changed=foo.c");
    cc::Build::new().file("foo.c").compile("foo");
}
