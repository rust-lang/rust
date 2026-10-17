//@ build-pass (FIXME(62277): could be check-pass?)

#![feature(type_alias_impl_trait)]

fn main() {}

// two definitions with different types
type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
fn foo() -> Foo {
    ""
}

#[define_opaque(Foo)]
fn bar(arg: bool) -> Foo {
    if arg {
        panic!()
    } else {
        "bar"
    }
}

#[define_opaque(Foo)]
fn boo(arg: bool) -> Foo {
    if arg {
        loop {}
    } else {
        "boo"
    }
}

#[define_opaque(Foo)]
fn bar2(arg: bool) -> Foo {
    if arg {
        "bar2"
    } else {
        panic!()
    }
}

#[define_opaque(Foo)]
fn boo2(arg: bool) -> Foo {
    if arg {
        "boo2"
    } else {
        loop {}
    }
}
