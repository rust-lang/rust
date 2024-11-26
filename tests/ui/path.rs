//@ run-pass

mod foo {
    pub fn bar(_offset: usize) { }
}

pub fn main() { foo::bar(0); }
