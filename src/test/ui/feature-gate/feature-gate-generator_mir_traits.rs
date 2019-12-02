// compile-fail
// edition:2018

fn is_send<T: Send>(val: T) {}

async fn dummy() {}

async fn not_send() {
    let val: *const ();
    dummy().await;
}

fn main() {
    is_send(not_send());
    //~^ ERROR  `*const ()` cannot be sent between threads safely
}
