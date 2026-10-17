//! regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/295

//@ compile-flags: -Znext-solver

#![forbid(unsafe_code)]

trait Trait<'a>: 'a {}

fn g<'s>(s: &'s String) -> &'static String
where
    dyn for<'x> Trait<'x> + 's: Trait<'static>,
{
    s
}

fn main() {
    let r = g(&String::from("freed"));
    //~^ ERROR temporary value dropped while borrowed
    println!("{r}");
}
