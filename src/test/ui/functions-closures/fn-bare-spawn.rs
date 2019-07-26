// run-pass
// This is what the signature to spawn should look like with bare functions


fn spawn<T:Send>(val: T, f: fn(T)) {
    f(val);
}

fn f(i: isize) {
    assert_eq!(i, 100);
}

pub fn main() {
    spawn(100, f);
}
