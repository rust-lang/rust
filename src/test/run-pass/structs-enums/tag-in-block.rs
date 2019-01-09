// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]



// pretty-expanded FIXME #23616

fn foo() {
    fn zed(_z: bar) { }
    enum bar { nil, }
    fn baz() { zed(bar::nil); }
}

pub fn main() { }
