// FIXME(fmease): Don't regress this.
//@ check-pass

trait Visit {}
impl<F> Visit for F where F: FnMut(&dyn Debug) {}

trait Debug {}

fn accept(_: &mut dyn Visit) {}

fn main() {
    // FIXME(fmease): This now fails with "implementation of `Fn*` is not general enough".
    accept(&mut |_: &dyn Debug| {});
}
