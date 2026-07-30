//@ check-pass

// A non-capturing closure whose fn-pointer signature is higher-ranked in two
// lifetimes coerces to a fn pointer that is higher-ranked in one (the outermost
// binder, at a coercion site). This exercises the closure coercion path.

fn main() {
    let c = |_: &(), _: &()| ();
    let _: for<'a> fn(&'a (), &'a ()) = c;
}
