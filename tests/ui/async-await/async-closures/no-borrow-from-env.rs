//@ edition: 2021
//@ check-pass

fn outlives<'a>(_: impl Sized + 'a) {}

async fn call_once(f: impl AsyncFnOnce()) {
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

    // outlives::<'a>(call_once(c));
    // The above fails b/c the by-move coroutine of `c` captures `x` in its entirety.
    // Since we have not asserted that the borrow for `&S<'a>` outlives `'a`, it'll fail.
}

fn main() {}
