// This is what the signature to spawn should look like with bare functions
#[legacy_modes];

fn spawn<T: Send>(val: T, f: extern fn(T)) {
    f(val);
}

fn f(&&i: int) {
    assert i == 100;
}

fn main() {
    spawn(100, f);
}
