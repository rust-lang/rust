// xfail-fast - check-fast doesn't understand aux-build
// aux-build:anon_trait_static_method_lib.rs

extern mod anon_trait_static_method_lib;
use anon_trait_static_method_lib::Foo;

fn main() {
    let x = Foo::new();
    io::println(x.x.to_str());
}



