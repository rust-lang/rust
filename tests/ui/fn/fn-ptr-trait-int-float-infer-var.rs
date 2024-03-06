//@ check-pass
trait MyCmp {
    fn cmp(&self) {}
}
impl MyCmp for f32 {}

fn main() {
    // Ensure that `impl<F: FnPtr> Ord for F` is never considered for int and float infer vars.
    0.0.cmp();
}
