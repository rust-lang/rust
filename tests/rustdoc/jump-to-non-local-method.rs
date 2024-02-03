// compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

use std::sync::atomic::AtomicIsize;

// @has 'src/foo/jump-to-non-local-method.rs.html'
// @has - '//a[@href="https://doc.rust-lang.org/nightly/core/sync/atomic/struct.AtomicIsize.html#method.new"]' 'AtomicIsize::new'

pub fn bar() {
    let _ = AtomicIsize::new(0);
    b();
}

fn b() {}
