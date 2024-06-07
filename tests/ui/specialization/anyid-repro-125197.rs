//@ aux-build: anyid-repro-125197.rs
//@ check-pass

// Makes sure that we don't check `specializes(impl1, impl2)` for a pair of impls that don't
// actually participate in specialization. Since <https://github.com/rust-lang/rust/pull/122791>,
// we don't treat inductive cycles as errors -- so we may need to winnow more pairs of impls, and
// we try to winnow impls in favor of other impls. However, if we're *inside* the `specializes`
// query, then may have a query cycle if we call `specializes` again!

extern crate anyid_repro_125197;
use anyid_repro_125197::AnyId;

fn main() {
    let x = "hello, world";
    let y: AnyId = x.into();
    let _ = y == x;
}
