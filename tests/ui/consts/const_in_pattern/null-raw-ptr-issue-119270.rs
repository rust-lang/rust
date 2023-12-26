// run-pass
// Eventually this will be rejected (when the future-compat lints are turned into hard errors), and
// then this test can be removed. But meanwhile we should ensure that this works and does not ICE.
struct NoDerive(i32);

#[derive(PartialEq)]
struct WrapEmbedded(*const NoDerive);

const WRAP_UNSAFE_EMBEDDED: &&WrapEmbedded = &&WrapEmbedded(std::ptr::null());

fn main() {
    let b = match WRAP_UNSAFE_EMBEDDED {
        WRAP_UNSAFE_EMBEDDED => true,
        //~^ WARN: must be annotated with `#[derive(PartialEq, Eq)]`
        //~| previously accepted
        _ => false,
    };
    assert!(b);
}
