// ignore-compare-mode-nll
// revisions: base nll
// [nll]compile-flags: -Zborrowck=mir

// check-fail
// See issue #91068. Types in the substs of an associated type can't be implied
// to be WF, since they don't actually have to be constructed.

trait Trait {
    type Type;
}

impl<T> Trait for T {
    type Type = ();
}

fn f<'a, 'b>(s: &'b str, _: <&'a &'b () as Trait>::Type) -> &'a str {
    s
    //[base]~^ ERROR lifetime mismatch [E0623]
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
    let x = String::from("Hello World!");
    let y = f(&x, ());
    drop(x);
    println!("{}", y);
}
