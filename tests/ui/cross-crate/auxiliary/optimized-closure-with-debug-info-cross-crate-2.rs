//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/31702
//@ compile-flags: -g

extern crate optimized_closure_with_debug_info_cross_crate_1;

use std::collections::HashMap;
use optimized_closure_with_debug_info_cross_crate_1::U256;

pub struct Ethash {
    engine_params: fn() -> Option<&'static Vec<u8>>,
    u256_params: HashMap<String, U256>,
}

impl Ethash {
    pub fn u256_param(&mut self, name: &str) -> U256 {
        let engine = self.engine_params;
        *self.u256_params.entry(name.to_owned()).or_insert_with(|| {
            engine().map_or(U256::new(0u64), |_a| loop {})
        })
    }
}
