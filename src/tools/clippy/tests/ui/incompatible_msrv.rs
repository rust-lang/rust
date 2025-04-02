#![warn(clippy::incompatible_msrv)]
#![feature(custom_inner_attributes)]
#![feature(panic_internals)]
#![clippy::msrv = "1.3.0"]

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::future::Future;
use std::thread::sleep;
use std::time::Duration;

fn foo() {
    let mut map: HashMap<&str, u32> = HashMap::new();
    assert_eq!(map.entry("poneyland").key(), &"poneyland");
    //~^ incompatible_msrv

    if let Entry::Vacant(v) = map.entry("poneyland") {
        v.into_key();
        //~^ incompatible_msrv
    }
    // Should warn for `sleep` but not for `Duration` (which was added in `1.3.0`).
    sleep(Duration::new(1, 0));
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
        core::panicking::panic("foo");
        //~^ ERROR: is `1.3.0` but this item is stable since `1.6.0`
    }

    // Lint code calling `core` from non-`core` macros
    macro_rules! my_panic {
        ($msg:expr) => {
            core::panicking::panic($msg)
        }; //~^ ERROR: is `1.3.0` but this item is stable since `1.6.0`
    }
    my_panic!("foo");

    // Lint even when the macro comes from `core` and calls `core` functions
    assert!(core::panicking::panic("out of luck"));
    //~^ ERROR: is `1.3.0` but this item is stable since `1.6.0`
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
    //~^ ERROR: is `1.80.0` but this item is stable since `1.82.0`
}

fn main() {}
