#![allow(unused)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::ptr_arg)] // https://github.com/rust-lang/rust-clippy/issues/10612
#![allow(clippy::needless_late_init)]
#![allow(clippy::box_collection)]
#![allow(clippy::boxed_local)]
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
    //~^ assigning_clones
}

fn clone_method_rhs_ref(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = ref_thing.clone();
    //~^ assigning_clones
}

fn clone_method_lhs_val(mut mut_thing: HasCloneFrom, ref_thing: &HasCloneFrom) {
    mut_thing = ref_thing.clone();
    //~^ assigning_clones
}

fn clone_function_lhs_mut_ref(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = Clone::clone(ref_thing);
    //~^ assigning_clones
}

fn clone_function_lhs_val(mut mut_thing: HasCloneFrom, ref_thing: &HasCloneFrom) {
    mut_thing = Clone::clone(ref_thing);
    //~^ assigning_clones
}

fn clone_function_through_trait(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = Clone::clone(ref_thing);
    //~^ assigning_clones
}

fn clone_function_through_type(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = HasCloneFrom::clone(ref_thing);
    //~^ assigning_clones
}

fn clone_function_fully_qualified(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    *mut_thing = <HasCloneFrom as Clone>::clone(ref_thing);
    //~^ assigning_clones
}

fn clone_method_lhs_complex(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    // These parens should be kept as necessary for a receiver
    *(mut_thing + &mut HasCloneFrom) = ref_thing.clone();
    //~^ assigning_clones
}

fn clone_method_rhs_complex(mut_thing: &mut HasCloneFrom, ref_thing: &HasCloneFrom) {
    // These parens should be removed since they are not needed in a function argument
    *mut_thing = (ref_thing + ref_thing).clone();
    //~^ assigning_clones
}

fn clone_method_macro() {
    let mut s = String::from("");
    s = format!("{} {}", "hello", "world").clone();
    //~^ assigning_clones
}

fn clone_function_macro() {
    let mut s = String::from("");
    s = Clone::clone(&format!("{} {}", "hello", "world"));
    //~^ assigning_clones
}

fn assign_to_init_mut_var(b: HasCloneFrom) -> HasCloneFrom {
    let mut a = HasCloneFrom;
    for _ in 1..10 {
        a = b.clone();
        //~^ assigning_clones
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

fn late_init_let_tuple() {
    let (p, q): (String, String);
    p = "ghi".to_string();
    q = p.clone();
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

#[clippy::msrv = "1.62"]
fn msrv_1_62(mut a: String, b: String, c: &str) {
    a = b.clone();
    //~^ assigning_clones
    // Should not be linted, as clone_into wasn't stabilized until 1.63
    a = c.to_owned();
}

#[clippy::msrv = "1.63"]
fn msrv_1_63(mut a: String, b: String, c: &str) {
    a = b.clone();
    //~^ assigning_clones
    a = c.to_owned();
    //~^ assigning_clones
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

// Make sure that we don't suggest the lint when we call clone inside a Clone impl
// https://github.com/rust-lang/rust-clippy/issues/12600
pub struct AvoidRecursiveCloneFrom;

impl Clone for AvoidRecursiveCloneFrom {
    fn clone(&self) -> Self {
        Self
    }
    fn clone_from(&mut self, source: &Self) {
        *self = source.clone();
    }
}

// Deref handling
fn clone_into_deref_method(mut a: DerefWrapper<HasCloneFrom>, b: HasCloneFrom) {
    *a = b.clone();
    //~^ assigning_clones
}

fn clone_into_deref_with_clone_method(mut a: DerefWrapperWithClone<HasCloneFrom>, b: HasCloneFrom) {
    *a = b.clone();
    //~^ assigning_clones
}

fn clone_into_box_method(mut a: Box<HasCloneFrom>, b: HasCloneFrom) {
    *a = b.clone();
    //~^ assigning_clones
}

fn clone_into_self_deref_method(a: &mut DerefWrapperWithClone<HasCloneFrom>, b: DerefWrapperWithClone<HasCloneFrom>) {
    *a = b.clone();
    //~^ assigning_clones
}

fn clone_into_deref_function(mut a: DerefWrapper<HasCloneFrom>, b: HasCloneFrom) {
    *a = Clone::clone(&b);
    //~^ assigning_clones
}

fn clone_into_box_function(mut a: Box<HasCloneFrom>, b: HasCloneFrom) {
    *a = Clone::clone(&b);
    //~^ assigning_clones
}

// ToOwned
fn owned_method_mut_ref(mut_string: &mut String, ref_str: &str) {
    *mut_string = ref_str.to_owned();
    //~^ assigning_clones
}

fn owned_method_val(mut mut_string: String, ref_str: &str) {
    mut_string = ref_str.to_owned();
    //~^ assigning_clones
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
    //~^ assigning_clones
}

fn owned_method_deref(mut_box_string: &mut HasDeref, ref_str: &str) {
    **mut_box_string = ref_str.to_owned();
    //~^ assigning_clones
}

fn owned_function_mut_ref(mut_thing: &mut String, ref_str: &str) {
    *mut_thing = ToOwned::to_owned(ref_str);
    //~^ assigning_clones
}

fn owned_function_val(mut mut_thing: String, ref_str: &str) {
    mut_thing = ToOwned::to_owned(ref_str);
    //~^ assigning_clones
}

fn owned_method_macro() {
    let mut s = String::from("");
    s = format!("{} {}", "hello", "world").to_owned();
    //~^ assigning_clones
}

fn owned_function_macro() {
    let mut s = String::from("");
    s = ToOwned::to_owned(&format!("{} {}", "hello", "world"));
    //~^ assigning_clones
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

mod borrowck_conflicts {
    //! Cases where clone_into and friends cannot be used because the src/dest have conflicting
    //! borrows.
    use std::path::PathBuf;

    fn issue12444(mut name: String) {
        let parts = name.split(", ").collect::<Vec<_>>();
        let first = *parts.first().unwrap();
        name = first.to_owned();
    }

    fn issue12444_simple() {
        let mut s = String::new();
        let s2 = &s;
        s = s2.to_owned();
    }

    fn issue12444_nodrop_projections() {
        struct NoDrop;

        impl Clone for NoDrop {
            fn clone(&self) -> Self {
                todo!()
            }
            fn clone_from(&mut self, other: &Self) {
                todo!()
            }
        }

        let mut s = NoDrop;
        let s2 = &s;
        s = s2.clone();

        let mut s = (NoDrop, NoDrop);
        let s2 = &s.0;
        s.0 = s2.clone();

        // This *could* emit a warning, but PossibleBorrowerMap only works with locals so it
        // considers `s` fully borrowed
        let mut s = (NoDrop, NoDrop);
        let s2 = &s.1;
        s.0 = s2.clone();
    }

    fn issue12460(mut name: String) {
        if let Some(stripped_name) = name.strip_prefix("baz-") {
            name = stripped_name.to_owned();
        }
    }

    fn issue12749() {
        let mut path = PathBuf::from("/a/b/c");
        path = path.components().as_path().to_owned();
    }
}

struct DerefWrapper<T>(T);

impl<T> Deref for DerefWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for DerefWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

struct DerefWrapperWithClone<T>(T);

impl<T> Deref for DerefWrapperWithClone<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for DerefWrapperWithClone<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Clone> Clone for DerefWrapperWithClone<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }

    fn clone_from(&mut self, source: &Self) {
        *self = Self(source.0.clone());
    }
}

#[cfg(test)]
mod test {
    #[derive(Default)]
    struct Data {
        field: String,
    }

    fn test_data() -> Data {
        Data {
            field: "default_value".to_string(),
        }
    }

    #[test]
    fn test() {
        let mut data = test_data();
        data.field = "override_value".to_owned();
    }
}
