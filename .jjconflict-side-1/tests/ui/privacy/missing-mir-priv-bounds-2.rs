// Test case from issue #151284.
// A private associated type bound allows to leak another private type and result in missing MIR.

//@ build-fail
//@ aux-crate:dep=missing-mir-priv-bounds-extern-2.rs

extern crate dep;
use dep::{GetUnreachable, PubTr, PubTrHandler, ToPriv, call_handler};

fn main() {
    call_handler::<Handler>();
}

struct Handler;
impl PubTrHandler for Handler {
    fn handle<T: PubTr>() {
        <T as Access>::AccessAssoc::generic::<i32>();
    }
}

trait Access: PubTr {
    type AccessAssoc;
}

impl<T: PubTr> Access for T {
    type AccessAssoc = <<T::Assoc as ToPriv>::AssocPriv as GetUnreachable>::Assoc;
}

//~? ERROR missing optimized MIR
