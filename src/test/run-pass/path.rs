// pretty-expanded FIXME #23616

mod foo {
    pub fn bar(_offset: usize) { }
}

pub fn main() { foo::bar(0); }
