// compile-flags: -Clink-dead-code=on --crate-type=lib
// build-pass

// Make sure that we don't monomorphize the impossible method `<() as Visit>::visit`,
// which does not hold under a reveal-all param env.

pub trait Visit {
    fn visit() {}
}

pub trait Array<'a> {}

impl Visit for () where (): for<'a> Array<'a> {}
