// run-pass
// needs-unwind
// ignore-wasm32-bare compiled with panic=abort by default
// revisions: default mir-opt
//[mir-opt] compile-flags: -Zmir-opt-level=4

#![allow(unconditional_panic)]

//! Test that panic locations for `#[track_caller]` functions in std have the correct
//! location reported.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::{Index, IndexMut};
use std::panic::{AssertUnwindSafe, UnwindSafe};

fn main() {
    // inspect the `PanicInfo` we receive to ensure the right file is the source
    std::panic::set_hook(Box::new(|info| {
        let actual = info.location().unwrap();
        if actual.file() != file!() {
            eprintln!("expected a location in the test file, found {:?}", actual);
            panic!();
        }
    }));

    fn assert_panicked(f: impl FnOnce() + UnwindSafe) {
        std::panic::catch_unwind(f).unwrap_err();
    }

    let nope: Option<()> = None;
    assert_panicked(|| nope.unwrap());
    assert_panicked(|| nope.expect(""));

    let oops: Result<(), ()> = Err(());
    assert_panicked(|| oops.unwrap());
    assert_panicked(|| oops.expect(""));

    let fine: Result<(), ()> = Ok(());
    assert_panicked(|| fine.unwrap_err());
    assert_panicked(|| fine.expect_err(""));

    let mut small = [0]; // the implementation backing str, vec, etc
    assert_panicked(move || { small.index(1); });
    assert_panicked(move || { small[1]; });
    assert_panicked(move || { small.index_mut(1); });
    assert_panicked(move || { small[1] += 1; });

    let sorted: BTreeMap<bool, bool> = Default::default();
    assert_panicked(|| { sorted.index(&false); });
    assert_panicked(|| { sorted[&false]; });

    let unsorted: HashMap<bool, bool> = Default::default();
    assert_panicked(|| { unsorted.index(&false); });
    assert_panicked(|| { unsorted[&false]; });

    let weirdo: VecDeque<()> = Default::default();
    assert_panicked(|| { weirdo.index(1); });
    assert_panicked(|| { weirdo[1]; });

    let refcell: RefCell<()> = Default::default();
    let _conflicting = refcell.borrow_mut();
    assert_panicked(AssertUnwindSafe(|| { refcell.borrow(); }));
    assert_panicked(AssertUnwindSafe(|| { refcell.borrow_mut(); }));
}
