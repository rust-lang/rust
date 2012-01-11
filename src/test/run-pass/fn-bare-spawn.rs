// This is what the signature to spawn should look like with bare functions

fn spawn<T: send>(val: T, f: native fn(T)) {
    f(val);
}

fn f(&&i: int) {
    assert i == 100;
}

fn main() {
    spawn(100, f);
}