//@compile-flags: --crate-type=lib -Clink-dead-code=on
//@ build-pass

// Make sure that we don't monomorphize the impossible method `<() as Visit>::visit`,
// which does not hold under a reveal-all param env.

pub trait Visit {
    fn visit() {}
}

pub trait Array {
    type Element;
}

impl<'a> Visit for () where (): Array<Element = &'a ()> {}

impl Array for () {
    type Element = ();
}
