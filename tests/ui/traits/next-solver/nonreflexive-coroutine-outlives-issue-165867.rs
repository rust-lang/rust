//@ edition: 2024
//@ compile-flags: -Znext-solver=globally
//@ check-fail

#![crate_type = "lib"]

use std::future::Future;

trait CoroutineOutlives<'a> {}

impl<'a, F> CoroutineOutlives<'a> for F where F: Future<Output = ()> + 'a {}

fn requires_every_lifetime<F: for<'a> CoroutineOutlives<'a>>(_: F) {}

pub fn cannot_extend_local_borrow() {
    let data = String::from("borrowed value");
    let short = &data; //~ ERROR E0597
    let future = async move {
        let _ = short.len();
    };
    requires_every_lifetime(future);
}
