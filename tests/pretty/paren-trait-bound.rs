//@ pp-exact

trait Dummy {}

// Without parens, `+ Send` would bind to `dyn Dummy` instead of the outer `dyn`.
fn f1(_: Box<dyn (Fn() -> Box<dyn Dummy>) + Send>) {}

// Without parens, `+ Send + Sync` would bind to `dyn Dummy` instead of the outer `impl`.
fn f2(_: impl (FnMut(&mut u8) -> &mut dyn Dummy) + Send + Sync) {}

fn main() {}
