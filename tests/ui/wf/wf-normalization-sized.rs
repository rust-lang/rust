//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-pass
//@[current] known-bug: #100041

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
//[next]~^ ERROR the size for values of type `[[[[[u8]]]]]` cannot be known at compilation time
const _: <Vec<str> as WellUnformed>::RequestNormalize = ();
//[next]~^ ERROR the size for values of type `str` cannot be known at compilation time

fn main() {}
