// run-pass
// ignore-wasm32-bare no env vars

use std::env::*;

fn main() {
    for (k, v) in vars_os() {
        let v2 = var_os(&k);
        assert!(v2.as_ref().map(|s| &**s) == Some(&*v),
                "bad vars->var transition: {:?} {:?} {:?}", k, v, v2);
    }
}
