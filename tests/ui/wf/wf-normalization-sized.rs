//@ check-pass
//@ known-bug: #100041

// Should fail. Normalization can bypass well-formedness checking.
// `[[[[[[u8]]]]]]` is not a well-formed type since size of type `[u8]` cannot
// be known at compile time (since `Sized` is not implemented for `[u8]`).

trait WellUnformed {
    type RequestNormalize;
}

impl<T: ?Sized> WellUnformed for T {
    type RequestNormalize = ();
}

const _: <[[[[[[u8]]]]]] as WellUnformed>::RequestNormalize = ();
const _: <Vec<str> as WellUnformed>::RequestNormalize = ();

fn main() {}
