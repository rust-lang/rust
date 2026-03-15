// Test for issue #153733: ICE with pin_ergonomics + async
//@compile-flags: --edition=2024
#![feature(min_generic_const_args)]
#![feature(adt_const_params)]
#![feature(pin_ergonomics)]

async fn other() {}

pub async fn uhoh() {
    other().await;
}

fn main() {}
