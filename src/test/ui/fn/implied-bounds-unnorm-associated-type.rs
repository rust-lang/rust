// check-fail
// See issue #91068. We check that the unnormalized associated types in
// function signatures are implied

trait Trait {
    type Type;
}

impl<T> Trait for T {
    type Type = ();
}

fn f<'a, 'b>(s: &'b str, _: <&'a &'b () as Trait>::Type) -> &'a str {
    //~^ ERROR in type `&'a &'b ()`, reference has a longer lifetime than the data it references
    s
}

fn main() {
    let x = String::from("Hello World!");
    let y = f(&x, ());
    drop(x);
    println!("{}", y);
}
