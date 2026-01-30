#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait<'a, 'b, 'c, A, B, C, const N: usize> {
    fn foo<'a, 'b, 'c, A, B, C, const N: usize>(&self) {
        //~^ ERROR: lifetime name `'a` shadows a lifetime name that is already in scope
        //~| ERROR: lifetime name `'b` shadows a lifetime name that is already in scope
        //~| ERROR: lifetime name `'c` shadows a lifetime name that is already in scope
        //~| ERROR: the name `A` is already used for a generic parameter in this item's generic parameters
        //~| ERROR: the name `B` is already used for a generic parameter in this item's generic parameters
        //~| ERROR: the name `C` is already used for a generic parameter in this item's generic parameters
        //~| ERROR: the name `N` is already used for a generic parameter in this item's generic parameters
    }
}

reuse Trait::foo;

fn main() {

}
