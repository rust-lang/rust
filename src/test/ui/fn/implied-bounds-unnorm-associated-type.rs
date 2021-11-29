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
    s //~ ERROR lifetime mismatch [E0623]
}

fn main() {
    let x = String::from("Hello World!");
    let y = f(&x, ());
    drop(x);
    println!("{}", y);
}
