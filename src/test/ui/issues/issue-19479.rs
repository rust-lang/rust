// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

trait Base {
    fn dummy(&self) { }
}
trait AssocA {
    type X: Base;
    fn dummy(&self) { }
}
trait AssocB {
    type Y: Base;
    fn dummy(&self) { }
}
impl<T: AssocA> AssocB for T {
    type Y = <T as AssocA>::X;
}

fn main() {}
