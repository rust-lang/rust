//@ check-pass

//! Check that borrowck accepts reborrowing types with multiple lifetimes when at
//! least one field is reborrowed, and types with a single lifetime and no
//! reborrowed field. See
//! `reborrow-multiple-lifetimes-no-reborrowed-field.rs` for the rejected cases.

#![feature(negative_impls, reborrow)]

use std::marker::{PhantomData, Reborrow};

struct Inner<'a>(&'a mut ());

impl<'a> Reborrow for Inner<'a> {}

struct WithReborrowedField<'a, 'b> {
    inner: Inner<'a>,
    shared: &'b (),
}

impl<'a, 'b> Reborrow for WithReborrowedField<'a, 'b> {}

struct WithMutField<'a, 'b> {
    mutable: &'a mut (),
    shared: &'b (),
}

impl<'a, 'b> Reborrow for WithMutField<'a, 'b> {}

struct SingleShared<'a> {
    shared: &'a (),
}

impl<'a> Reborrow for SingleShared<'a> {}

// `NoCopyInner` has no `Copy` impl, so `inner` is reborrowed in every body, even
// one with a `T: Copy` bound.
struct NoCopyInner<'a, T>(&'a mut (), PhantomData<T>);

impl<'a, T> Reborrow for NoCopyInner<'a, T> {}

struct WithNoCopyField<'a, 'b, T> {
    inner: NoCopyInner<'a, T>,
    shared: &'b (),
}

impl<'a, 'b, T> Reborrow for WithNoCopyField<'a, 'b, T> {}

fn use_no_copy_field<'a, 'b, T: Copy>(x: WithNoCopyField<'a, 'b, T>) -> WithNoCopyField<'a, 'b, T> {
    x
}

// A negative `Copy` impl does not count as a `Copy` impl, so `inner` still counts.
struct NotCopyMarker<'a>(PhantomData<&'a mut ()>);

impl<'a> Reborrow for NotCopyMarker<'a> {}
impl<'a> !Copy for NotCopyMarker<'a> {}

struct WithNotCopyMarker<'a, 'b> {
    inner: NotCopyMarker<'a>,
    shared: &'b (),
}

impl<'a, 'b> Reborrow for WithNotCopyMarker<'a, 'b> {}

fn use_not_copy_marker<'a, 'b>(x: WithNotCopyMarker<'a, 'b>) -> WithNotCopyMarker<'a, 'b> {
    x
}

fn use_reborrowed_field<'a, 'b>(x: WithReborrowedField<'a, 'b>) -> WithReborrowedField<'a, 'b> {
    x
}

fn use_mut_field<'a, 'b>(x: WithMutField<'a, 'b>) -> WithMutField<'a, 'b> {
    x
}

fn use_single_shared<'a>(x: SingleShared<'a>) -> SingleShared<'a> {
    x
}

// Each value is used again after the first reborrow ends, which a plain move would reject.
fn main() {
    let mut a = ();
    let b = ();

    let reborrowed = WithReborrowedField { inner: Inner(&mut a), shared: &b };
    let _ = use_reborrowed_field(reborrowed);
    let _ = use_reborrowed_field(reborrowed);

    let mutable = WithMutField { mutable: &mut a, shared: &b };
    let _ = use_mut_field(mutable);
    let _ = use_mut_field(mutable);

    let single = SingleShared { shared: &b };
    let _ = use_single_shared(single);
    let _ = use_single_shared(single);
}
