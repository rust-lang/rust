// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

trait Common { fn dummy(&self) { } }

impl<'t, T> Common for (T, &'t T) {}

impl<'t, T> Common for (&'t T, T) {}

fn main() {}
