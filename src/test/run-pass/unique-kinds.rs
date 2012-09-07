use cmp::Eq;

fn sendable() {

    fn f<T: Send Eq>(i: T, j: T) {
        assert i == j;
    }

    fn g<T: Send Eq>(i: T, j: T) {
        assert i != j;
    }

    let i = ~100;
    let j = ~100;
    f(i, j);
    let i = ~100;
    let j = ~101;
    g(i, j);
}

fn copyable() {

    fn f<T: Copy Eq>(i: T, j: T) {
        assert i == j;
    }

    fn g<T: Copy Eq>(i: T, j: T) {
        assert i != j;
    }

    let i = ~100;
    let j = ~100;
    f(i, j);
    let i = ~100;
    let j = ~101;
    g(i, j);
}

fn noncopyable() {

    fn f<T: Eq>(i: T, j: T) {
        assert i == j;
    }

    fn g<T: Eq>(i: T, j: T) {
        assert i != j;
    }

    let i = ~100;
    let j = ~100;
    f(i, j);
    let i = ~100;
    let j = ~101;
    g(i, j);
}

fn main() {
    sendable();
    copyable();
    noncopyable();
}
