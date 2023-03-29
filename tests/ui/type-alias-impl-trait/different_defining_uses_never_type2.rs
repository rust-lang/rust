// check-pass

#![feature(type_alias_impl_trait)]

fn main() {}

// all defininions have the same type
type Foo = impl std::fmt::Debug;

#[defines(Foo)]
fn foo() -> Foo {
    ""
}

#[defines(Foo)]
fn bar(arg: bool) -> Foo {
    if arg { panic!() } else { "bar" }
}

#[defines(Foo)]
fn boo(arg: bool) -> Foo {
    if arg { loop {} } else { "boo" }
}

#[defines(Foo)]
fn bar2(arg: bool) -> Foo {
    if arg { "bar2" } else { panic!() }
}

#[defines(Foo)]
fn boo2(arg: bool) -> Foo {
    if arg { "boo2" } else { loop {} }
}
