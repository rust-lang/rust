//@ compile-flags: -Z public-api-hash

#![crate_name = "dep"]
#![crate_type = "rlib"]

// `#[inline]` makes this function cross-crate-inlinable, so its MIR is encoded into the
// rmeta. That MIR references `private::helper`, which makes `helper` reachable in the
// public-API reachability graph even though it is private.
#[inline]
pub fn call_private() {
    helper();
}

#[cfg(any(cpass1))]
fn helper() {}

#[cfg(any(cpass2))]
fn helper() {}
