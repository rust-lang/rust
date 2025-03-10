// Combination of `ptr-to-trait-obj-different-regions-misc.rs` and `ptr-to-trait-obj-wrap.rs`.
//
// Checks that you *can't* change lifetime arguments of trait objects in pointer casts involving
// wrapping said traits structures.

trait A<'a> {}

struct W<T: ?Sized>(T);
struct X<T: ?Sized>(T);

fn unwrap<'a, 'b>(a: *const W<dyn A<'a>>) -> *const dyn A<'b> {
    a as _
    //~^ error
    //~| error
}

fn unwrap_nested<'a, 'b>(a: *const W<W<dyn A<'a>>>) -> *const W<dyn A<'b>> {
    a as _
    //~^ error
    //~| error
}

fn rewrap<'a, 'b>(a: *const W<dyn A<'a>>) -> *const X<dyn A<'b>> {
    a as _
    //~^ error: lifetime may not live long enough
    //~| error: lifetime may not live long enough
}

fn rewrap_nested<'a, 'b>(a: *const W<W<dyn A<'a>>>) -> *const W<X<dyn A<'b>>> {
    a as _
    //~^ error: lifetime may not live long enough
    //~| error: lifetime may not live long enough
}

fn wrap<'a, 'b>(a: *const dyn A<'a>) -> *const W<dyn A<'b>> {
    a as _
    //~^ error: lifetime may not live long enough
    //~| error: lifetime may not live long enough
}

fn main() {}
