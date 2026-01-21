//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ aux-build: dependency.rs
//@ ignore-backends: gcc

// This test verifies SVH (Strict Version Hash) stability behavior:
//
// rpass1 -> rpass2: Only a PRIVATE function body changes in the dependency.
//   The SVH should NOT change, so this crate should be fully reused.
//
// rpass2 -> rpass3: An INLINABLE (#[inline]) function body changes.
//   The SVH SHOULD change, so this crate needs to be re-codegened.

#![feature(rustc_attrs)]

// When private function changes (rpass1 -> rpass2): ALL modules should be REUSED
// because the SVH doesn't change when only private items change.
#![rustc_partition_reused(module="main-use_public", cfg="rpass2")]
#![rustc_partition_reused(module="main-use_inlinable", cfg="rpass2")]
#![rustc_partition_reused(module="main-use_struct", cfg="rpass2")]

// When inlinable function changes (rpass2 -> rpass3): only the module that
// uses the inlinable function should be CODEGENED. Other modules are REUSED.
#![rustc_partition_reused(module="main-use_public", cfg="rpass3")]
#![rustc_partition_codegened(module="main-use_inlinable", cfg="rpass3")]
#![rustc_partition_reused(module="main-use_struct", cfg="rpass3")]

extern crate dependency;

pub mod use_public {
    use dependency::public_function;

    pub fn call_public() -> i32 {
        public_function()
    }
}

pub mod use_inlinable {
    use dependency::inlinable_function;

    pub fn call_inlinable() -> i32 {
        inlinable_function()
    }
}

pub mod use_struct {
    use dependency::Data;

    pub fn make_data() -> Data {
        Data::new(42)
    }
}

fn main() {
    let _ = use_public::call_public();
    let _ = use_inlinable::call_inlinable();
    let _ = use_struct::make_data();
}
