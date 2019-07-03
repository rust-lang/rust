// build-pass (FIXME(62277): could be check-pass?)

// Test that we propagate region relations from closures precisely when there is
// more than one non-local lower bound.

// In this case the closure has signature
// |x: &'4 mut (&'5 (&'1 str, &'2 str), &'3 str)| -> ..
// We end up with a `'3: '5` constraint that we can propagate as
// `'3: '1`, `'3: '2`, but previously we approximated it as `'3: 'static`.

// As an optimization, we primarily propagate bounds for the "representative"
// of each SCC. As such we have these two similar cases where hopefully one
// of them will test the case we want (case2, when this test was added).
mod case1 {
    fn f(s: &str) {
        g(s, |x| h(x));
    }

    fn g<T, F>(_: T, _: F)
    where F: Fn(&mut (&(T, T), T)) {}

    fn h<T>(_: &mut (&(T, T), T)) {}
}

mod case2 {
    fn f(s: &str) {
        g(s, |x| h(x));
    }

    fn g<T, F>(_: T, _: F)
    where F: Fn(&mut (T, &(T, T))) {}

    fn h<T>(_: &mut (T, &(T, T))) {}
}

fn main() {}
