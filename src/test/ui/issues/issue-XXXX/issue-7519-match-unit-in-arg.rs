// run-pass
// pretty-expanded FIXME #23616

/*
#7519 ICE pattern matching unit in function argument
*/

fn foo(():()) { }

pub fn main() {
    foo(());
}
