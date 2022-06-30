#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> Foo;

fn foo() -> Foo {
    foo //~ ERROR: overflow evaluating the requirement `fn() -> Foo {foo}: Sized`
}

fn main() {}
