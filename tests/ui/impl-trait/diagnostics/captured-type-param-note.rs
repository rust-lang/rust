//@ edition: 2024
//@ dont-require-annotations: NOTE

fn assert_static<T: 'static>(_: T) {}

fn bar<T>(_: T) -> impl Iterator<Item = u8> {
    std::iter::once(0)
}

fn foo(a: &str) {
    assert_static(bar(a.split(',')));
    //~^ ERROR: borrowed data escapes outside of function
    //~| NOTE: the `impl Iterator<Item = u8>` returned by `bar` captures `T`
}

fn tuple(a: &str) {
    assert_static((bar(a.split(',')), 1u8));
    //~^ ERROR: borrowed data escapes outside of function
    //~| NOTE: the `impl Iterator<Item = u8>` returned by `bar` captures `T`
}

fn nested(a: &str) {
    assert_static(bar(bar(a.split(','))));
    //~^ ERROR: borrowed data escapes outside of function
    //~| NOTE: the `impl Iterator<Item = u8>` returned by `bar` captures `T`
}

fn bar2<T, U>(_: T, _: U) -> impl Iterator<Item = u8> {
    std::iter::once(0)
}

fn capture_t(a: &str) {
    assert_static(bar2(a.split(','), 5u8));
    //~^ ERROR: borrowed data escapes outside of function
    //~| NOTE: the `impl Iterator<Item = u8>` returned by `bar2` captures `T`
}

fn capture_u(a: &str) {
    assert_static(bar2(5u8, a.split(',')));
    //~^ ERROR: borrowed data escapes outside of function
    //~| NOTE: the `impl Iterator<Item = u8>` returned by `bar2` captures `U`
}

async fn bar3<T>(_: T) {}

fn foo_async(a: &str) {
    assert_static(bar3(a.split(',')));
    //~^ ERROR: borrowed data escapes outside of function
    //~| NOTE: the `impl Future<Output = ()>` returned by `bar3` captures `T`
}

struct S<U>(U);

impl<U> S<U> {
    fn m<T>(self, _: T) -> impl Iterator<Item = u8> {
        std::iter::once(0)
    }
}

fn method_param(a: &str) {
    assert_static(S(5u8).m(a.split(',')));
    //~^ ERROR: borrowed data escapes outside of function
    //~| NOTE: the `impl Iterator<Item = u8>` returned by `m` captures `T`
}

fn impl_param(a: &str) {
    assert_static(S(a.split(',')).m(5u8));
    //~^ ERROR: borrowed data escapes outside of function
    //~| NOTE: the `impl Iterator<Item = u8>` returned by `m` captures `U`
}

// Note will not appear in these cases below

fn tuple_no_error_on_bar(a: &str) {
    assert_static((bar(5u8), a.split(',')));
    //~^ ERROR: borrowed data escapes outside of function
}

fn bar_not_shortest_path(a: &str) {
    assert_static((bar(a.split(',')), a.split(',')));
    //~^ ERROR: borrowed data escapes outside of function
}

fn f<T>(_: T, _: &mut Vec<T>) -> impl Sized {}

fn not_related_to_impl_trait_err(a: &str) {
    let mut out = Vec::new();
    let _ = f(a.split(','), &mut out);
    assert_static(out);
    //~^ ERROR: borrowed data escapes outside of function
}

fn main() {}
