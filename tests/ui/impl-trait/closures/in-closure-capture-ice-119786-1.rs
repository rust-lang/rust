//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #119786
//@ edition:2021

#![feature(type_alias_impl_trait)]

fn enum_upvar() {
    type T = impl Copy;
    let foo: T = Some((1u32, 2u32));
    let x = move || {
        match foo {
            None => (),
            Some(yield) => (),
            //[next]~^ ERROR: expected identifier, found reserved keyword `yield`
        }
    };
}

pub fn main() {}
