//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// We have to prove implied bounds of higher-ranked types at some point.
// Normalizing associated types can drop requirements. This means we need
// to prove well-formedness when normalizing them, at least as long as
// implied bounds are implicit.

trait Trait {
    type Assoc<'a, 'b: 'a>;
}

impl Trait for () {
    type Assoc<'a, 'b: 'a> = ();
}

fn foo<'a, 'b, T: Trait>(_: <T as Trait>::Assoc<'a, 'b>, x: &'b str) -> &'a str {
    x
}

fn main() {
    let func: for<'a, 'b> fn((), &'b str) -> &'static str = foo::<()>;
    //[current]~^ ERROR higher-ranked lifetime error
    //[next]~^^ ERROR mismatched types
    let x: &'static str = func((), &String::from("temporary"));
    println!("{x}");
}
