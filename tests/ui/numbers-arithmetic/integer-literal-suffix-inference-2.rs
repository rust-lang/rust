//@ run-pass

fn foo(_: *const ()) {}

fn main() {
    let a = 3;
    foo(&a as *const _ as *const ());
}
