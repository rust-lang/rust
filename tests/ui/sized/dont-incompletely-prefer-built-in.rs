//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

struct W<T: ?Sized>(T);

fn is_sized<T: Sized>(x: *const T) {}

fn dummy<T: ?Sized>() -> *const T { todo!() }

fn non_param_where_bound<T: ?Sized>()
where
    W<T>: Sized,
{
    let x: *const W<_> = dummy();
    is_sized::<W<_>>(x);
    let _: *const W<T> = x;
}

fn main() {}
