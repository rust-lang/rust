//! This test checks that opaque type collection doesn't try to normalize the projection
//! without respecting its binders (which would ICE).
//! Unfortunately we don't even reach opaque type collection, as we ICE in typeck before that.
// known-bug: #109281
// failure-status: 101
// error-pattern:internal compiler error
// normalize-stderr-test "internal compiler error.*" -> ""
// normalize-stderr-test "DefId\([^)]*\)" -> "..."
// normalize-stderr-test "\nerror: internal compiler error.*\n\n" -> ""
// normalize-stderr-test "note:.*unexpectedly panicked.*\n\n" -> ""
// normalize-stderr-test "note: we would appreciate a bug report.*\n\n" -> ""
// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*\n\n" -> ""
// normalize-stderr-test "thread.*panicked.*\n" -> ""
// normalize-stderr-test "stack backtrace:\n" -> ""
// normalize-stderr-test "\s\d{1,}: .*\n" -> ""
// normalize-stderr-test "\s at .*\n" -> ""
// normalize-stderr-test ".*note: Some details.*\n" -> ""
// normalize-stderr-test "\n\n[ ]*\n" -> ""
// normalize-stderr-test "compiler/.*: projection" -> "projection"
// edition:2018

#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]

use std::future::Future;

struct Foo<'a>(&'a mut ());

type Fut<'a> = impl Future<Output = ()>;

trait Trait<'x> {
    type Thing;
}

impl<'x, T: 'x> Trait<'x> for (T,) {
    type Thing = T;
}

impl Foo<'_> {
    fn make_fut(&self) -> Box<dyn for<'a> Trait<'a, Thing = Fut<'a>>> {
        Box::new((async { () },))
    }
}

fn main() {}
