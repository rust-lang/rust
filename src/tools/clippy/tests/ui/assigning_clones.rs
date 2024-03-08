#![allow(unused)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::ptr_arg)] // https://github.com/rust-lang/rust-clippy/issues/10612
#![allow(clippy::needless_late_init)]
#![allow(clippy::box_collection)]
#![warn(clippy::assigning_clones)]

use std::borrow::ToOwned;
use std::ops::{Add, Deref, DerefMut};

// Clone
pub struct HasCloneFrom;

impl Clone for HasCloneFrom {
    fn clone(&self) -> Self {
        Self
    }
    fn clone_from(&mut self, source: &Self) {
        *self = HasCloneFrom;
    }
}

fn clone_method_rhs_val(mut_thing: &mut HasCloneFrom, value_thing: HasCloneFrom) {
    *mut_thing = value_thing.clone();
}

fn clone_method_rhs_ref(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = ref_thing.clone();
}

fn clone_method_lhs_val(mut mut_thing: HasCloneFrom, ref_thing: &HasCloneFrom) {
    mut_thing = ref_thing.clone();
}

fn clone_function_lhs_mut_ref(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = Clone::clone(ref_thing);
}

fn clone_function_lhs_val(mut mut_thing: HasCloneFrom, ref_thing: &HasCloneFrom) {
    mut_thing = Clone::clone(ref_thing);
}

fn clone_function_through_trait(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = Clone::clone(ref_thing);
}

fn clone_function_through_type(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = HasCloneFrom::clone(ref_thing);
}

fn clone_function_fully_qualified(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = <HasCloneFrom as Clone>::clone(ref_thing);
}

fn clone_method_lhs_complex(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    // These parens should be kept as necessary for a receiver
    *(mut_thing + &mut HasCloneFrom) = ref_thing.clone();
}

fn clone_method_rhs_complex(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    // These parens should be removed since they are not needed in a function argument
    *mut_thing = (ref_thing + ref_thing).clone();
}

fn assign_to_init_mut_var(b: HasCloneFrom) -> HasCloneFrom {
    let mut a = HasCloneFrom;
    for _ in 1..10 {
        a = b.clone();
    }
    a
}

fn assign_to_late_init_mut_var(b: HasCloneFrom) {
    let mut a;
    a = HasCloneFrom;
    a = b.clone();
}

fn assign_to_uninit_var(b: HasCloneFrom) {
    let a;
    a = b.clone();
}

fn assign_to_uninit_mut_var(b: HasCloneFrom) {
    let mut a;
    a = b.clone();
}

#[derive(Clone)]
pub struct HasDeriveClone;

fn ignore_derive_clone(a: &mut HasDeriveClone, b: &HasDeriveClone) {
    // Should not be linted, since the Clone impl is derived
    *a = b.clone();
}

pub struct HasCloneImpl;

impl Clone for HasCloneImpl {
    fn clone(&self) -> Self {
        Self
    }
}

fn ignore_missing_clone_from(a: &mut HasCloneImpl, b: &HasCloneImpl) {
    // Should not be linted, since the Clone impl doesn't override clone_from
    *a = b.clone();
}

struct FakeClone;

impl FakeClone {
    /// This looks just like `Clone::clone`
    fn clone(&self) -> Self {
        FakeClone
    }
}

fn ignore_fake_clone() {
    let mut a = FakeClone;
    let b = FakeClone;
    // Should not be linted, since the Clone impl doesn't come from std
    a = b.clone();
}

fn ignore_generic_clone<T: Clone>(a: &mut T, b: &T) {
    // Should not be linted, since we don't know the actual clone impl
    *a = b.clone();
}

macro_rules! clone_inside {
    ($a:expr, $b: expr) => {
        $a = $b.clone();
    };
}

fn clone_inside_macro() {
    let mut a = String::new();
    let b = String::new();
    clone_inside!(a, b);
}

// ToOwned
fn owned_method_mut_ref(mut_string: &mut String, ref_str: &str) {
    *mut_string = ref_str.to_owned();
}

fn owned_method_val(mut mut_string: String, ref_str: &str) {
    mut_string = ref_str.to_owned();
}

struct HasDeref {
    a: String,
}

impl Deref for HasDeref {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.a
    }
}

impl DerefMut for HasDeref {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.a
    }
}

fn owned_method_box(mut_box_string: &mut Box<String>, ref_str: &str) {
    **mut_box_string = ref_str.to_owned();
}

fn owned_method_deref(mut_box_string: &mut HasDeref, ref_str: &str) {
    **mut_box_string = ref_str.to_owned();
}

fn owned_function_mut_ref(mut_thing: &mut String, ref_str: &str) {
    *mut_thing = ToOwned::to_owned(ref_str);
}

fn owned_function_val(mut mut_thing: String, ref_str: &str) {
    mut_thing = ToOwned::to_owned(ref_str);
}

struct FakeToOwned;
impl FakeToOwned {
    /// This looks just like `ToOwned::to_owned`
    fn to_owned(&self) -> Self {
        FakeToOwned
    }
}

fn fake_to_owned() {
    let mut a = FakeToOwned;
    let b = FakeToOwned;
    // Should not be linted, since the ToOwned impl doesn't come from std
    a = b.to_owned();
}

fn main() {}

/// Trait implementation to allow producing a `Thing` with a low-precedence expression.
impl Add for HasCloneFrom {
    type Output = Self;
    fn add(self, _: HasCloneFrom) -> Self {
        self
    }
}
/// Trait implementation to allow producing a `&Thing` with a low-precedence expression.
impl<'a> Add for &'a HasCloneFrom {
    type Output = Self;
    fn add(self, _: &'a HasCloneFrom) -> Self {
        self
    }
}
/// Trait implementation to allow producing a `&mut Thing` with a low-precedence expression.
impl<'a> Add for &'a mut HasCloneFrom {
    type Output = Self;
    fn add(self, _: &'a mut HasCloneFrom) -> Self {
        self
    }
}
