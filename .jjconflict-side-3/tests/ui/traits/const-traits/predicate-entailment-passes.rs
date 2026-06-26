//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

const trait Bar {}
const impl Bar for () {}

const trait TildeConst {
    fn foo<T>() where T: [const] Bar;
}
impl TildeConst for () {
    fn foo<T>() where T: Bar {}
}


const trait AlwaysConst {
    fn foo<T>() where T: const Bar;
}
impl AlwaysConst for i32 {
    fn foo<T>() where T: Bar {}
}
const impl AlwaysConst for u32 {
    fn foo<T>() where T: [const] Bar {}
}

fn main() {}
