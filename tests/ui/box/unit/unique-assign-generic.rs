//@ run-pass

fn f<T>(t: T) -> T {
    let t1 = t;
    t1
}

pub fn main() {
    let t = f::<Box<_>>(Box::new(100));
    assert_eq!(t, Box::new(100));
}
