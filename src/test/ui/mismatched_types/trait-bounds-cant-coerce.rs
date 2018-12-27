trait Foo {
    fn dummy(&self) { }
}

fn a(_x: Box<Foo+Send>) {
}

fn c(x: Box<Foo+Sync+Send>) {
    a(x);
}

fn d(x: Box<Foo>) {
    a(x); //~ ERROR mismatched types [E0308]
}

fn main() { }
