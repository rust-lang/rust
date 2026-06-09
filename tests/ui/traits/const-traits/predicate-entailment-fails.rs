//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

const trait Bar {}
impl const Bar for () {}


const trait TildeConst {
    type Bar<T> where T: [const] Bar;

    fn foo<T>() where T: [const] Bar;
}
impl TildeConst for () {
    type Bar<T> = () where T: const Bar;
    //~^ ERROR impl has stricter requirements than trait

    fn foo<T>() where T: const Bar {}
    //~^ ERROR impl has stricter requirements than trait
}


const trait NeverConst {
    type Bar<T> where T: Bar;

    fn foo<T>() where T: Bar;
}
impl NeverConst for i32 {
    type Bar<T> = () where T: const Bar;
    //~^ ERROR impl has stricter requirements than trait

    fn foo<T>() where T: const Bar {}
    //~^ ERROR impl has stricter requirements than trait
}
impl const NeverConst for u32 {
    type Bar<T> = () where T: [const] Bar;
    //~^ ERROR impl has stricter requirements than trait

    fn foo<T>() where T: [const] Bar {}
    //~^ ERROR impl has stricter requirements than trait
}

fn main() {}
