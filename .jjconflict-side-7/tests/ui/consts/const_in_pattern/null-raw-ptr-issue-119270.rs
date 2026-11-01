//@ run-pass
struct NoDerive(#[allow(dead_code)] i32);

#[derive(PartialEq)]
struct WrapEmbedded(*const NoDerive);

const WRAP_UNSAFE_EMBEDDED: &&WrapEmbedded = &&WrapEmbedded(std::ptr::null());

fn main() {
    let b = match WRAP_UNSAFE_EMBEDDED {
        WRAP_UNSAFE_EMBEDDED => true,
        _ => false,
    };
    assert!(b);
}
