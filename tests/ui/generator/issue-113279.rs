#![feature(generators)]

// `foo` attempts to dereference `""`, which results in an error being reported. Later, the
// generator transform for `foo` then produces a union which contains a `str` type - unions should
// not contain unsized types, but this is okay because an error has been reported already.
// When const propagation happens later in compilation, it attempts to compute the layout of the
// generator (as part of checking whether something can be const propagated) and in turn attempts
// to compute the layout of `str` in the context of a union - where this caused an ICE. This test
// makes sure that doesn't happen again.

fn foo() {
    let _y = static || {
        let x = &mut 0;
        *{
            yield;
            x
        } += match { *"" }.len() {
            //~^ ERROR cannot move a value of type `str` [E0161]
            //~^^ ERROR cannot move out of a shared reference [E0507]
            _ => 0,
        };
    };
}

fn main() {
    foo()
}
