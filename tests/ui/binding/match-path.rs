// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


// pretty-expanded FIXME #23616

mod m1 {
    pub enum foo { foo1, foo2, }
}

fn bar(x: m1::foo) { match x { m1::foo::foo1 => { } m1::foo::foo2 => { } } }

pub fn main() { }
