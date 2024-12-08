//@ check-pass

trait A<X> {
    fn dummy(&self, arg: X);
}

trait B {
    type X;
    type Y: A<Self::X>;

    fn dummy(&self);
}

fn main () { }
