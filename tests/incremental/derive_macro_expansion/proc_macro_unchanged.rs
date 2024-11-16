// This test tests that derive-macro execution is cached.
// HOWEVER, this test can currently only be checked manually,
// by running it (through compiletest) with `-- --nocapture --verbose`.
// The proc-macro (for `Nothing`) prints a message to stderr when invoked,
// and this message should only be present during the first invocation,
// because the cached result should be used for the second invocation.
// FIXME(pr-time): Properly have the test check this, but how? UI-test that tests for `.stderr`?

//@ aux-build:derive_nothing.rs
//@ revisions:cfail1 cfail2
//@ compile-flags: -Z query-dep-graph -Zcache-all-derive-macros=true
//@ build-pass

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(dead_code)]
#![crate_type = "rlib"]

#![rustc_partition_codegened(module="proc_macro_unchanged-foo", cfg="cfail1")]
// #![rustc_partition_codegened(module="proc_macro_unchanged-foo", cfg="cfail2")]

// `foo::nothing_mod` is created by the derive macro and doesn't change
// BUG: this yields the same result with `-Zcache-all-derive-macros=false` (i.e., uncached),
// not sure how to do this correctly.
#![rustc_partition_reused(module="proc_macro_unchanged-foo-nothing_mod", cfg="cfail2")]

 #[macro_use]
 extern crate derive_nothing;

pub mod foo {
    #[derive(Nothing)]
    pub struct Foo;

    pub fn use_foo(_f: Foo) {
        nothing_mod::nothing();

        eprintln!("foo used");
    }
}
