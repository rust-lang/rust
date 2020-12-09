// revisions:rpass1 rpass2 rpass3
// compile-flags: -Z query-dep-graph -g -O
// aux-build:extern_crate.rs

// ignore-asmjs wasm2js does not support source maps yet

// This test case makes sure that we detect if paths emitted into debuginfo
// are changed, even when the change happens in an external crate.

// NOTE: We're explicitly passing the `-O` optimization flag because if no optimizations are
// requested, rustc will ignore the `#[inline]` attribute. This is a performance optimization for
// non-optimized builds which causes us to generate fewer copies of inlined functions when
// runtime performance doesn't matter. Without this flag, the function will go into a different
// CGU which can be reused by this crate.

#![feature(rustc_attrs)]

#![rustc_partition_reused(module="main", cfg="rpass2")]
#![rustc_partition_reused(module="main-some_mod", cfg="rpass2")]
#![rustc_partition_reused(module="main", cfg="rpass3")]
#![rustc_partition_codegened(module="main-some_mod", cfg="rpass3")]

extern crate extern_crate;

fn main() {
    some_mod::some_fn();
}

mod some_mod {
    use extern_crate;

    pub fn some_fn() {
        extern_crate::inline_fn();
    }
}
