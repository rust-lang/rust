// Check that signature comparison for tail calls allows subtyping.
//
//@ check-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn f<'short>(
    a: &'static (),
    b: &'short (),
    c: fn(&'static ()),
    d: for<'a> fn(&'a ()),
) -> (&'short (), fn(&'static ()), for<'a> fn(&'a ())) {
    become g(b, a, d, c);
}

fn g<'short>(
    // swapped short/static
    a: &'short (),
    b: &'static (),
    // swapped binder/non binder
    c: for<'a> fn(&'a ()),
    d: fn(&'static ()),
    // 'short=>'static; non binder=>binder
) -> (&'static (), for<'a> fn(&'a ()), for<'a> fn(&'a ())) {
    (b, c, c)
}

fn main() {}
