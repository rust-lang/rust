// edition:2018
// revisions:rpass1
#![feature(const_generics)]

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
