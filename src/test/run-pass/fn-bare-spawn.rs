// This is what the signature to spawn should look like with bare functions

fn spawn<~T>(val: T, f: fn(T)) {
    f(val);
}

fn f(&&i: int) {
    assert i == 100;
}

fn main() {
    spawn(100, f);
    spawn(100, fn(&&i: int) {
        assert i == 100;
    });
}