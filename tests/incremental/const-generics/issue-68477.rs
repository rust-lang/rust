//@ edition:2018
//@ revisions:rpass1

// Needed to supply generic arguments to the anon const in `[(); FOO]`.
#![feature(generic_const_exprs)]

const FOO: usize = 1;

struct Container<T> {
    val: std::marker::PhantomData<T>,
    blah: [(); FOO]
}

async fn dummy() {}

async fn foo() {
    let a: Container<&'static ()>;
    dummy().await;
}

fn is_send<T: Send>(_: T) {}

fn main() {
    is_send(foo());
}
