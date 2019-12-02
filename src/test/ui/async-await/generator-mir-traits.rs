// edition:2018
// check-pass

#![feature(generator_mir_traits)]

fn is_send<T: Send>(val: T) {}

async fn dummy() {}

async fn not_send() {
    let val: *const ();
    dummy().await;
}

fn main() {
    is_send(not_send());
}
