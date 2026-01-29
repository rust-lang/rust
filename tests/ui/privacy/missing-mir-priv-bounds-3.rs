// Test case from issue #151479.
// A private associated type bound allows to leak another private type and result in missing MIR.

//@ build-fail
//@ aux-crate:dep=missing-mir-priv-bounds-extern-3.rs

extern crate dep;
use dep::{GetUnreachable, Sub, SubHandler, Super, call_handler};

fn main() {
    call_handler::<Handler>();
}

struct Handler;
impl SubHandler for Handler {
    fn handle<T: Sub>() {
        <T as Access>::AccessAssoc::generic::<i32>();
    }
}

// Without this indirection, Handler::handle notices that
// it's mentioning dep::Priv.
trait Access: Super {
    type AccessAssoc;
}
impl<T: Super> Access for T {
    type AccessAssoc = <<T as Super>::AssocSuper as GetUnreachable>::Assoc;
}

//~? ERROR missing optimized MIR
