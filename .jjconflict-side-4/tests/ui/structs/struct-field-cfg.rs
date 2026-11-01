struct Foo {
    present: (),
}

fn main() {
    let foo = Foo { #[cfg(true)] present: () };
    let _ = Foo { #[cfg(false)] present: () };
    //~^ ERROR missing field `present` in initializer of `Foo`
    let _ = Foo { present: (), #[cfg(false)] absent: () };
    let _ = Foo { present: (), #[cfg(true)] absent: () };
    //~^ ERROR struct `Foo` has no field named `absent`
    let Foo { #[cfg(true)] present: () } = foo;
    let Foo { #[cfg(false)] present: () } = foo;
    //~^ ERROR pattern does not mention field `present`
    let Foo { present: (), #[cfg(false)] absent: () } = foo;
    let Foo { present: (), #[cfg(true)] absent: () } = foo;
    //~^ ERROR struct `Foo` does not have a field named `absent`
}
