// Regression test for #61315
//
// `dyn T:` is lowered to `dyn T: ReEmpty` - check that we don't ICE in NLL for
// the unexpected region.

// build-pass (FIXME(62277): could be check-pass?)

trait T {}
fn f() where dyn T: {}

fn main() {}
