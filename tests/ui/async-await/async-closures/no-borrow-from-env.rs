//@ edition: 2021
//@ check-pass

#![feature(async_closure)]

fn outlives<'a>(_: impl Sized + 'a) {}

async fn call_once(f: impl async FnOnce()) {
    f().await;
}

fn simple<'a>(x: &'a i32) {
    let c = async || { println!("{}", *x); };
    outlives::<'a>(c());
    outlives::<'a>(call_once(c));

    let c = async move || { println!("{}", *x); };
    outlives::<'a>(c());
    outlives::<'a>(call_once(c));
}

struct S<'a>(&'a i32);

fn through_field<'a>(x: S<'a>) {
    let c = async || { println!("{}", *x.0); };
    outlives::<'a>(c());
    outlives::<'a>(call_once(c));

    let c = async move || { println!("{}", *x.0); };
    outlives::<'a>(c());
    outlives::<'a>(call_once(c));
}

fn through_field_and_ref<'a>(x: &S<'a>) {
    let c = async || { println!("{}", *x.0); };
    outlives::<'a>(c());
    outlives::<'a>(call_once(c));

    let c = async move || { println!("{}", *x.0); };
    outlives::<'a>(c());
    // outlives::<'a>(call_once(c)); // FIXME(async_closures): Figure out why this fails
}

fn main() {}
