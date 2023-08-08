// compile-flags: -Ztrait-solver=next
// check-pass

// Makes sure we don't prepopulate the MIR typeck of `define`
// with `Foo<T, U> = T`, but instead, `Foo<B, A> = B`, so that
// the param-env predicates actually apply.

#![feature(type_alias_impl_trait)]

type Foo<T: Send, U> = impl NeedsSend<T>;

trait NeedsSend<T> {}
impl<T: Send> NeedsSend<T> for T {}

fn define<A, B: Send>(a: A, b: B, _: Foo<B, A>) {
    let y: Option<Foo<B, A>> = Some(b);
}

fn main() {}
