#![warn(clippy::incompatible_msrv)]
#![feature(custom_inner_attributes)]
#![allow(stable_features)]
#![feature(strict_provenance)] // For use in test
#![clippy::msrv = "1.3.0"]

use std::cell::Cell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::future::Future;
use std::thread::sleep;
use std::time::Duration;

fn foo() {
    let mut map: HashMap<&str, u32> = HashMap::new();
    assert_eq!(map.entry("poneyland").key(), &"poneyland");
    //~^ incompatible_msrv
    //~| NOTE: `-D clippy::incompatible-msrv` implied by `-D warnings`
    //~| HELP: to override `-D warnings` add `#[allow(clippy::incompatible_msrv)]`

    if let Entry::Vacant(v) = map.entry("poneyland") {
        v.into_key();
        //~^ incompatible_msrv
    }
    // Should warn for `sleep` but not for `Duration` (which was added in `1.3.0`).
    sleep(Duration::new(1, 0));
    //~^ incompatible_msrv
}

#[clippy::msrv = "1.2.0"]
static NO_BODY_BAD_MSRV: Option<Duration> = None;
//~^ incompatible_msrv

static NO_BODY_GOOD_MSRV: Option<Duration> = None;

#[clippy::msrv = "1.2.0"]
fn bad_type_msrv() {
    let _: Option<Duration> = None;
    //~^ incompatible_msrv
}

#[test]
fn test() {
    sleep(Duration::new(1, 0));
}

#[clippy::msrv = "1.63.0"]
async fn issue12273(v: impl Future<Output = ()>) {
    // `.await` desugaring has a call to `IntoFuture::into_future` marked #[stable(since = "1.64.0")],
    // but its stability is ignored
    v.await;
}

fn core_special_treatment(p: bool) {
    // Do not lint code coming from `core` macros expanding into `core` function calls
    if p {
        panic!("foo"); // Do not lint
    }

    // But still lint code calling `core` functions directly
    if p {
        let _ = core::iter::once_with(|| 0);
        //~^ incompatible_msrv
    }

    // Lint code calling `core` from non-`core` macros
    macro_rules! my_panic {
        ($msg:expr) => {
            let _ = core::iter::once_with(|| $msg);
            //~^ incompatible_msrv
        };
    }
    my_panic!("foo");

    // Lint even when the macro comes from `core` and calls `core` functions
    assert!(core::iter::once_with(|| 0).next().is_some());
    //~^ incompatible_msrv
}

#[clippy::msrv = "1.26.0"]
fn lang_items() {
    // Do not lint lang items. `..=` will expand into `RangeInclusive::new()`, which was introduced
    // in Rust 1.27.0.
    let _ = 1..=3;
}

#[clippy::msrv = "1.80.0"]
fn issue14212() {
    let _ = std::iter::repeat_n((), 5);
    //~^ incompatible_msrv
}

#[clippy::msrv = "1.0.0"]
fn cstr_and_cstring_ok() {
    let _: Option<&'static std::ffi::CStr> = None;
    let _: Option<std::ffi::CString> = None;
}

fn local_msrv_change_suggestion() {
    let _ = std::iter::repeat_n((), 5);
    //~^ incompatible_msrv

    #[cfg(any(test, not(test)))]
    {
        let _ = std::iter::repeat_n((), 5);
        //~^ incompatible_msrv
        //~| NOTE: you may want to conditionally increase the MSRV

        // Emit the additional note only once
        let _ = std::iter::repeat_n((), 5);
        //~^ incompatible_msrv
    }
}

#[clippy::msrv = "1.78.0"]
fn feature_enable_14425(ptr: *const u8) -> usize {
    // Do not warn, because it is enabled through a feature even though
    // it is stabilized only since Rust 1.84.0.
    let r = ptr.addr();

    // Warn about this which has been introduced in the same Rust version
    // but is not allowed through a feature.
    r.isqrt()
    //~^ incompatible_msrv
}

fn non_fn_items() {
    let _ = std::io::ErrorKind::CrossesDevices;
    //~^ incompatible_msrv
}

#[clippy::msrv = "1.87.0"]
fn msrv_non_ok_in_const() {
    {
        let c = Cell::new(42);
        _ = c.get();
    }
    const {
        let c = Cell::new(42);
        _ = c.get();
        //~^ incompatible_msrv
    }
}

#[clippy::msrv = "1.88.0"]
fn msrv_ok_in_const() {
    {
        let c = Cell::new(42);
        _ = c.get();
    }
    const {
        let c = Cell::new(42);
        _ = c.get();
    }
}

#[clippy::msrv = "1.86.0"]
fn enum_variant_not_ok() {
    let _ = std::io::ErrorKind::InvalidFilename;
    //~^ incompatible_msrv
    let _ = const { std::io::ErrorKind::InvalidFilename };
    //~^ incompatible_msrv
}

#[clippy::msrv = "1.87.0"]
fn enum_variant_ok() {
    let _ = std::io::ErrorKind::InvalidFilename;
    let _ = const { std::io::ErrorKind::InvalidFilename };
}

fn main() {}
