// aux-build:extern-crate.rs
#![feature(rustc_attrs)]
extern crate extern_crate;

impl extern_crate::StructWithAttr { //~ ERROR
    fn foo() {}
}
impl extern_crate::StructWithAttr {
    #[rustc_allow_incoherent_impl]
    fn bar() {}
}
impl extern_crate::StructNoAttr { //~ ERROR
    fn foo() {}
}
impl extern_crate::StructNoAttr { //~ ERROR
    #[rustc_allow_incoherent_impl]
    fn bar() {}
}
impl extern_crate::EnumWithAttr { //~ ERROR
    fn foo() {}
}
impl extern_crate::EnumWithAttr {
    #[rustc_allow_incoherent_impl]
    fn bar() {}
}
impl extern_crate::EnumNoAttr { //~ ERROR
    fn foo() {}
}
impl extern_crate::EnumNoAttr { //~ ERROR
    #[rustc_allow_incoherent_impl]
    fn bar() {}
}

fn main() {}
