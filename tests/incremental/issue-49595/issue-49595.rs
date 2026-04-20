//@ revisions: bfail1 bfail2 bfail3
//@ compile-flags: -Z query-dep-graph --test
//@ build-pass
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "rlib"]

#![rustc_partition_codegened(module="issue_49595-tests", cfg="bfail2")]
#![rustc_partition_codegened(module="issue_49595-lit_test", cfg="bfail3")]

mod tests {
    #[cfg_attr(not(bfail1), test)]
    fn _test() {
    }
}


// Checks that changing a string literal without changing its span
// takes effect.

// replacing a module to have a stable span
#[cfg_attr(not(bfail3), path = "auxiliary/lit_a.rs")]
#[cfg_attr(bfail3, path = "auxiliary/lit_b.rs")]
mod lit;

pub mod lit_test {
    #[test]
    fn lit_test() {
        println!("{}", ::lit::A);
    }
}
