//@ aux-build:derive_nothing.rs
//@ revisions:cfail1 cfail2
//@ compile-flags: -Z query-dep-graph
//@ check-pass (FIXME(62277): could be check-pass?)

// TODO(pr-time): do these revisions make sense? only "check" required?

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(dead_code)]
#![crate_type = "rlib"]

#![rustc_partition_codegened(module="item_changed-foo", cfg="cfail1")]
// #![rustc_partition_reused(module="item_changed-foo", cfg="cfail2")]
#![rustc_partition_reused(module="item_changed-foo-nothing_mod", cfg="cfail2")]

 #[macro_use]
 extern crate derive_nothing;

pub mod foo {
    // #[rustc_clean(cfg="cfail2")]
     #[derive(Nothing)]
    pub struct Foo;

    #[cfg(cfail2)]
    pub fn second_fn() {
        eprintln!("just forcing codegen");
    }

    pub fn use_foo(_f: Foo) {
        // #[cfg(cfail1)]
        nothing_mod::nothing();

        // #[cfg(cfail2)]
        // nothingx();

        eprintln!("foo used");
    }
}

// fn main() {
//     Foo;
//
//     nothing();
//
//     #[cfg(rpass2)]
//     Bar;
// }
