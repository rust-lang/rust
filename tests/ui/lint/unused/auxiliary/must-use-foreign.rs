//@ edition:2021

use std::future::Future;

pub struct Manager;

impl Manager {
    #[must_use]
    pub async fn new() -> (Self, impl Future<Output = ()>) {
        (Manager, async {})
    }
}
