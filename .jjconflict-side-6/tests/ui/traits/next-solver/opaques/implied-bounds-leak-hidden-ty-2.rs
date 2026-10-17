//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#159. We need to make sure
// that computing the implied assumptions of `wat` does not look into the hidden
// type `impl Sized`, as doing so adds a `T: 'static` implied bound which
// its caller does not have to prove.

fn into_y<T>(t: T) -> impl Sized
where
    T: 'static,
{
    t
}
fn wat<T>(t: T) -> impl Sized + 'static {
    //[next]~^ ERROR the parameter type `T` may not live long enough
    into_y(t) //~ ERROR the parameter type `T` may not live long enough
}

fn leak<T>(t: &T) -> &'static T {
    *(&wat(t) as &dyn std::any::Any).downcast_ref().unwrap()
}

fn main() {
    let buf = leak(&vec![vec![1]]);
    dbg!(buf[0][0]);
}
