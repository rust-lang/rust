// Combination of `ptr-to-trait-obj-different-args.rs` and `ptr-to-trait-obj-wrap.rs`.
//
// Checks that you *can't* change type arguments of trait objects in pointer casts involving
// wrapping said traits structures.

trait A<T> {}

struct W<T: ?Sized>(T);
struct X<T: ?Sized>(T);

fn unwrap<F, G>(a: *const W<dyn A<F>>) -> *const dyn A<G> {
    a as _
    //~^ error casting `*const W<(dyn A<F> + 'static)>` as `*const dyn A<G>` is invalid
}

fn unwrap_nested<F, G>(a: *const W<W<dyn A<F>>>) -> *const W<dyn A<G>> {
    a as _
    //~^ error casting `*const W<W<(dyn A<F> + 'static)>>` as `*const W<dyn A<G>>` is invalid
}

fn rewrap<F, G>(a: *const W<dyn A<F>>) -> *const X<dyn A<G>> {
    a as _
    //~^ error: casting `*const W<(dyn A<F> + 'static)>` as `*const X<dyn A<G>>` is invalid
}

fn rewrap_nested<F, G>(a: *const W<W<dyn A<F>>>) -> *const W<X<dyn A<G>>> {
    a as _
    //~^ error: casting `*const W<W<(dyn A<F> + 'static)>>` as `*const W<X<dyn A<G>>>` is invalid
}

fn wrap<F, G>(a: *const dyn A<F>) -> *const W<dyn A<G>> {
    a as _
    //~^ error: casting `*const (dyn A<F> + 'static)` as `*const W<dyn A<G>>` is invalid
}

fn main() {}
