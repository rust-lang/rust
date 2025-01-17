//@ check-pass

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
