//@ known-bug: #120503
#![feature(effects)]

trait MyTrait {}

impl MyTrait for i32 {
    async const fn bar(&self) {
        main8().await;
    }
}
