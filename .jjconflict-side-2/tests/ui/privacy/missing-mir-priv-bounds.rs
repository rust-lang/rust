// Test case from issue #151284.
// A private associated type bound allows to leak another private type and result in missing MIR.

//@ build-fail
//@ aux-crate:dep=missing-mir-priv-bounds-extern.rs

extern crate dep;
use dep::{GetUnreachable, PubTr, ToPriv, get_dummy};

fn main() {
    wut(get_dummy());
}

fn wut<T: PubTr>(_: T) {
    <T as Access>::AccessAssoc::generic::<i32>();
}

trait Access: PubTr {
    type AccessAssoc;
}

impl<T: PubTr> Access for T {
    type AccessAssoc = <<T::Assoc as ToPriv>::AssocPriv as GetUnreachable>::Assoc;
}

//~? ERROR missing optimized MIR
