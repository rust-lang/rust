//@ compile-flags: -g

extern crate issue_31702_1;

use std::collections::HashMap;
use issue_31702_1::U256;

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
