//@ check-pass
pub trait Handler {
    fn handle(&self, _: &mut String);
}

impl<F> Handler for F where F: for<'a, 'b> Fn(&'a mut String) {
    fn handle(&self, st: &mut String) {
        self(st)
    }
}

fn main() {}
