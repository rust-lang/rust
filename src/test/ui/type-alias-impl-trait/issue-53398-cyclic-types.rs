#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> Foo;

fn foo() -> Foo {
//~^ ERROR: overflow evaluating the requirement `fn() -> Foo {foo}: Sized`
    foo
}

fn main() {}
