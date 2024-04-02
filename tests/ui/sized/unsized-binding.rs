struct Foo<T: ?Sized>(i32, T);
trait T {}

fn bar() -> dyn T { //~ ERROR E0746
    panic!()
}

fn main() {
    let x = *""; //~ ERROR E0277
    println!("{}", x);
    println!("{}", x);
    let Foo(a, b) = Foo(1, bar());
    //~^ ERROR E0277
    //~| ERROR E0277
    // The second error above could be deduplicated if we remove or unify the additional `note`
    // explaining unsized locals and fn arguments.
}
