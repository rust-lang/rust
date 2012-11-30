// xfail-test
// aux-build:private_variant_1.rs
extern mod private_variant_1;

fn main() {
    let _x = private_variant_1::super_sekrit::baz; //~ ERROR baz is private
}