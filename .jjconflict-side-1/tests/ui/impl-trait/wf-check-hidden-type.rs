//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

//! Regression test for #114728. This also catched
//! trait-system-refactor-initiative#159 with the new
//! solver.

trait Extend<'a, 'b> {
    fn extend(self, _: &'a str) -> &'b str;
}

impl<'a, 'b> Extend<'a, 'b> for Option<&'b &'a ()> {
    fn extend(self, s: &'a str) -> &'b str {
        s
    }
}

fn boom<'a, 'b>() -> impl Extend<'a, 'b> {
    //[next]~^ ERROR lifetime may not live long enough
    None::<&'_ &'_ ()> //[current]~ ERROR lifetime may not live long enough
}

fn main() {
    let y = boom().extend(&String::from("temporary"));
    println!("{}", y);
}
