// xfail-fast
// aux-build:xc_private_method_lib.rs

extern mod xc_private_method_lib;

fn main() {
    // normal method on struct
    let _ = xc_private_method_lib::Struct{ x: 10 }.meth_struct();  //~ ERROR method `meth_struct` is private
    // static method on struct
    let _ = xc_private_method_lib::Struct::static_meth_struct();  //~ ERROR function `static_meth_struct` is private

    // normal method on enum
    let _ = xc_private_method_lib::Variant1(20).meth_enum();  //~ ERROR method `meth_enum` is private
    // static method on enum
    let _ = xc_private_method_lib::Enum::static_meth_enum();  //~ ERROR function `static_meth_enum` is private
}
