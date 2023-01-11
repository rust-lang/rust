// run-pass
// ignore-wasm32-bare no env vars

use std::env::*;

fn main() {
    for (k, v) in vars_os() {
        // On Windows, the environment variable NUMBER_OF_PROCESSORS has special meaning.
        // Unfortunately, you can get different answers, depending on whether you are
        // enumerating all environment variables or querying a specific variable.
        // This was causing this test to fail on machines with more than 64 processors.
        if cfg!(target_os = "windows") && k == "NUMBER_OF_PROCESSORS" {
            continue;
        }

        let v2 = var_os(&k);
        assert!(v2.as_ref().map(|s| &**s) == Some(&*v),
                "bad vars->var transition: {:?} {:?} {:?}", k, v, v2);
    }
}
