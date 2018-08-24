struct Foo {
    present: (),
}

fn main() {
    let foo = Foo { #[cfg(all())] present: () };
    let _ = Foo { #[cfg(any())] present: () };
    //~^ ERROR missing field `present` in initializer of `Foo`
    let _ = Foo { present: (), #[cfg(any())] absent: () };
    let _ = Foo { present: (), #[cfg(all())] absent: () };
    //~^ ERROR struct `Foo` has no field named `absent`
    let Foo { #[cfg(all())] present: () } = foo;
    let Foo { #[cfg(any())] present: () } = foo;
    //~^ ERROR pattern does not mention field `present`
    let Foo { present: (), #[cfg(any())] absent: () } = foo;
    let Foo { present: (), #[cfg(all())] absent: () } = foo;
    //~^ ERROR struct `Foo` does not have a field named `absent`
}
