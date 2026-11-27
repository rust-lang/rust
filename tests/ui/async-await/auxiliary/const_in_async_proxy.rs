//@ edition:2024

#![feature(async_drop)]
#![allow(incomplete_features)]
#![allow(long_running_const_eval)]

pub async fn has_const() {
    const CONST: () = {
        let mut i = 0;

        while i < 10_000_000 {
            i += 1;
        }
    };

    CONST
}
