//@ run-pass

fn foo(x: &mut Box<u8>) {
    *x = Box::new(5);
}

pub fn main() {
    foo(&mut Box::new(4));
}
