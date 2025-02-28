//@ check-pass

// Ensure that a param-env predicate like `(T,): Sized` doesn't get in the way of us
// confirming the built-in call operator for `AsyncFnOnce`. This happens because we
// end up with a signature like:
//
// `<F as AsyncFnOnce<(?0,)>>::async_call_once(self) -> <F as AsyncFnOnce<(?0,)>>::CallOnceFuture`
//
// (where we use fresh infer vars for each arg since we haven't actually typeck'd the args yet).
// But normalizing that signature keeps the associated type rigid, so we don't end up
// constraining `?0` like we would if we were normalizing the analogous `FnOnce` call...
// Then we were checking that the method signature was WF, which would incompletely constrain
// `(?0,) == (T,)` via the param-env, leading to us later failing on `F: AsyncFnOnce<(T,)>`.

fn run<F, T>(f: F)
where
    F: AsyncFnOnce(i32),
    (T,): Sized,
{
    f(1i32);
}

fn main() {}
