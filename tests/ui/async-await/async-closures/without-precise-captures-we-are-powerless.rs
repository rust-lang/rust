//@ edition: 2018

// This is `no-borrow-from-env.rs`, but under edition 2018 we still want to make
// sure that we don't ICE or anything, even if precise closure captures means
// that we can't actually borrowck successfully.

fn outlives<'a>(_: impl Sized + 'a) {}

async fn call_once(f: impl AsyncFnOnce()) {
    f().await;
}

fn simple<'a>(x: &'a i32) {
    let c = async || { println!("{}", *x); }; //~ ERROR `x` does not live long enough
    outlives::<'a>(c());
    outlives::<'a>(call_once(c));

    let c = async move || { println!("{}", *x); };
    outlives::<'a>(c()); //~ ERROR `c` does not live long enough
    outlives::<'a>(call_once(c));
}

struct S<'a>(&'a i32);

fn through_field<'a>(x: S<'a>) {
    let c = async || { println!("{}", *x.0); }; //~ ERROR `x` does not live long enough
    outlives::<'a>(c());
    outlives::<'a>(call_once(c));

    let c = async move || { println!("{}", *x.0); }; //~ ERROR cannot move out of `x`
    outlives::<'a>(c()); //~ ERROR `c` does not live long enough
    outlives::<'a>(call_once(c)); //~ ERROR cannot move out of `c`
}

fn through_field_and_ref<'a>(x: &S<'a>) {
    let c = async || { println!("{}", *x.0); }; //~ ERROR `x` does not live long enough
    outlives::<'a>(c());
    outlives::<'a>(call_once(c)); //~ ERROR explicit lifetime required in the type of `x`
}

fn through_field_and_ref_move<'a>(x: &S<'a>) {
    let c = async move || { println!("{}", *x.0); };
    outlives::<'a>(c()); //~ ERROR `c` does not live long enough
    outlives::<'a>(call_once(c)); //~ ERROR explicit lifetime required in the type of `x`
}

struct T;
impl T {
    fn outlives<'a>(&'a self, _: impl Sized + 'a) {}
}
fn through_method<'a>(x: &'a i32) {
    let c = async || { println!("{}", *x); }; //~ ERROR `x` does not live long enough
    T.outlives::<'a>(c());
    T.outlives::<'a>(call_once(c));

    let c = async move || { println!("{}", *x); };
    T.outlives::<'a>(c()); //~ ERROR `c` does not live long enough
    T.outlives::<'a>(call_once(c));
}

fn main() {}
