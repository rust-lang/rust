// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

trait A {
    fn a_method(&self);
}

trait B: A {
    fn b_method(&self);
}

trait C: B {
    fn c_method(&self) {
        self.a_method();
    }
}

pub fn main() {}
