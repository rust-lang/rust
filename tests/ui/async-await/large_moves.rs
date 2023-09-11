#![deny(large_assignments)]
#![cfg_attr(attribute, feature(large_assignments))]
#![cfg_attr(attribute, move_size_limit = "1000")]
// build-fail
// only-x86_64
// revisions: attribute option nothing
// [option]compile-flags: -Zmove-size-limit=2000

// edition:2018
// compile-flags: -Zmir-opt-level=0

use std::{sync::Arc, rc::Rc};

fn main() {
    let x = async {
        let y = [0; 9999];
        dbg!(y);
        thing(&y).await;
        dbg!(y);
    };
    let z = (x, 42); //~ ERROR large_assignments
    let a = z.0; //~ ERROR large_assignments
    let b = z.1;
    let _ = Arc::new([0; 9999]); // OK!
    let _ = Box::new([0; 9999]); // OK!
    let _ = Rc::new([0; 9999]); // OK!
    let _ = NotBox::new([0; 9999]); //~ ERROR large_assignments
    default_limits();
}

fn default_limits() {
    // Moving 500 bytes is OK in all revisions
    let _ = NotBox::new([0; 500]);

    // Moving 1500 bytes should fail only with the `attribute` revision because
    // its limit is 1000 bytes
    let _ = NotBox::new([0; 1500]); //[attribute]~ ERROR large_assignments

    // Moving 2500 bytes should fail with both `attribute` and `option` since
    // their limits are 1000 and 2000 respectively.
    let _ = NotBox::new([0; 2500]);
    //[attribute]~^ ERROR large_assignments
    //[option]~^^ ERROR large_assignments

    // With a nightly compiler the default limit is 4096. So 5000 should fail
    // for all revisions
    let _ = NotBox::new([0; 5000]); //~ ERROR large_assignments
}

async fn thing(y: &[u8]) {
    dbg!(y);
}

struct NotBox<const N: usize> {
    data: [u8; N],
}

impl<const N: usize> NotBox<N> {
    fn new(data: [u8; N]) -> Self {
        // FIXME: Each different instantiation of this generic type (with
        // different N) results in a unique error message. Deduplicate somehow.
        Self {
            data, //~ ERROR large_assignments
                  //[nothing]~^ ERROR large_assignments
                  //[option]~| ERROR large_assignments
                  //[option]~| ERROR large_assignments
                  //[attribute]~| ERROR large_assignments
                  //[attribute]~| ERROR large_assignments
                  //[attribute]~| ERROR large_assignments
        }
    }
}
