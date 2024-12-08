#![feature(never_type)]
#![allow(
    unused_mut,
    clippy::redundant_allocation,
    clippy::needless_pass_by_ref_mut,
    static_mut_refs
)]
#![warn(clippy::must_use_candidate)]
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct MyAtomic(AtomicBool);
pub struct MyPure;

pub fn pure(i: u8) -> u8 {
    i
}

impl MyPure {
    pub fn inherent_pure(&self) -> u8 {
        0
    }
}

pub trait MyPureTrait {
    fn trait_pure(&self, i: u32) -> u32 {
        self.trait_impl_pure(i) + 1
    }

    fn trait_impl_pure(&self, i: u32) -> u32;
}

impl MyPureTrait for MyPure {
    fn trait_impl_pure(&self, i: u32) -> u32 {
        i
    }
}

pub fn without_result() {
    // OK
}

pub fn impure_primitive(i: &mut u8) -> u8 {
    *i
}

pub fn with_callback<F: Fn(u32) -> bool>(f: &F) -> bool {
    f(0)
}

pub fn with_marker(_d: std::marker::PhantomData<&mut u32>) -> bool {
    true
}

pub fn quoth_the_raven(_more: !) -> u32 {
    unimplemented!();
}

pub fn atomics(b: &AtomicBool) -> bool {
    b.load(Ordering::SeqCst)
}

pub fn rcd(_x: Rc<u32>) -> bool {
    true
}

pub fn rcmut(_x: Rc<&mut u32>) -> bool {
    true
}

pub fn arcd(_x: Arc<u32>) -> bool {
    false
}

pub fn inner_types(_m: &MyAtomic) -> bool {
    true
}

static mut COUNTER: usize = 0;

/// # Safety
///
/// Don't ever call this from multiple threads
pub unsafe fn mutates_static() -> usize {
    COUNTER += 1;
    COUNTER
}

#[no_mangle]
pub extern "C" fn unmangled(i: bool) -> bool {
    !i
}

fn main() {
    assert_eq!(1, pure(1));
}
