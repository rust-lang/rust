// Incorrect handling of uninhabited types could cause us to mark coroutine
// types as entirely uninhabited, when they were in fact constructible. This
// caused us to hit "unreachable" code (illegal instruction on x86).

//@ run-pass

//@ compile-flags: -Aunused
//@ edition: 2018

pub enum Uninhabited { }

fn uninhabited_async() -> Uninhabited {
    unreachable!()
}

async fn noop() { }

async fn contains_never() {
    let error = uninhabited_async();
    noop().await;
    let error2 = error;
}

async fn overlap_never() {
    let error1 = uninhabited_async();
    noop().await;
    let error2 = uninhabited_async();
    drop(error1);
    noop().await;
    drop(error2);
}

#[allow(unused_must_use)]
fn main() {
}
