//@ check-pass

#![feature(cfg_eval)]

struct Value;

impl Value {
    fn method(self) -> usize {
        0
    }
}

fn main() {
    let _ = #[cfg_eval] #[inline] || {};
    let _ = #[cfg_eval] #[allow(unused)] Value.method();
}
