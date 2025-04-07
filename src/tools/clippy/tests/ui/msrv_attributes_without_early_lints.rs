//@check-pass

#![allow(clippy::all, clippy::pedantic, clippy::restriction, clippy::nursery)]
#![forbid(clippy::ptr_as_ptr)]

/// MSRV checking in late passes skips checking the parent nodes if no early pass sees a
/// `#[clippy::msrv]` attribute
///
/// Here we ensure that even if all early passes are allowed (above) the attribute is still detected
/// in late lints such as `clippy::ptr_as_ptr`
#[clippy::msrv = "1.37"]
fn f(p: *const i32) {
    let _ = p as *const i64;
}
