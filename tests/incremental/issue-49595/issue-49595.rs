//@ revisions: bpass1 bpass2 bpass3
//@ compile-flags: -Z query-dep-graph --test
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "rlib"]

#![rustc_partition_codegened(module="issue_49595-tests", cfg="bpass2")]
#![rustc_partition_codegened(module="issue_49595-lit_test", cfg="bpass3")]

mod tests {
    #[cfg_attr(not(bpass1), test)]
    fn _test() {
    }
}


// Checks that changing a string literal without changing its span
// takes effect.

// replacing a module to have a stable span
#[cfg_attr(not(bpass3), path = "auxiliary/lit_a.rs")]
#[cfg_attr(bpass3, path = "auxiliary/lit_b.rs")]
mod lit;

pub mod lit_test {
    #[test]
    fn lit_test() {
        println!("{}", ::lit::A);
    }
}
