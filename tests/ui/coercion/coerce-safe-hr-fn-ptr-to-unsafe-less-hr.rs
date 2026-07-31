//@ check-pass

// A *safe*, higher-ranked function pointer coerces to an *unsafe*,
// less-higher-ranked function pointer at a coercion site. This combines the
// higher-ranked -> less-higher-ranked step (`Adjust::Subtype`) with the
// safe -> unsafe step (`UnsafeFnPointer`); the target is still higher-ranked,
// so the coercion must enter the target's binder as placeholders.

fn main() {
    let f: for<'a, 'b> fn(&'a (), &'b ()) = |_, _| ();
    let _: for<'a> unsafe fn(&'a (), &'a ()) = f;
}
