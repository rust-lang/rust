// pretty-expanded FIXME #23616

#![allow(unreachable_code)]

fn foo() -> bool { false }

fn bar() {
    return;
    !foo();
}

fn baz() {
    return;
    if "" == "" {}
}

pub fn main() {
    bar();
    baz();
}
