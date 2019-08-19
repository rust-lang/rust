// Check that the hash of `foo` doesn't change just because we ordered
// the nested items (or even added new ones).

// revisions: cfail1 cfail2
// build-pass (FIXME(62277): could be check-pass?)

#![crate_type = "rlib"]
#![feature(rustc_attrs)]

#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
pub fn foo() {
    #[cfg(cfail1)]
    pub fn baz() { } // order is different...

    #[rustc_clean(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="HirBody", cfg="cfail2")]
    pub fn bar() { } // but that doesn't matter.

    #[cfg(cfail2)]
    pub fn baz() { } // order is different...

    pub fn bap() { } // neither does adding a new item
}
