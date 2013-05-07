// xfail-fast
// aux-build:xc_private_method_lib.rs

extern mod xc_private_method_lib;

fn main() {
    let _ = xc_private_method_lib::Foo::new();  //~ ERROR function `new` is private
}
