// run-pass

fn foo(x: &usize) -> usize {
    *x
}

pub fn main() {
    let p: Box<_> = Box::new(22);
    let r = foo(&*p);
    println!("r={}", r);
    assert_eq!(r, 22);
}
