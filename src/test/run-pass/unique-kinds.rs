use cmp::Eq;

fn sendable() {

    fn f<T: send Eq>(i: T, j: T) {
        assert i == j;
    }

    fn g<T: send Eq>(i: T, j: T) {
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

    fn f<T: copy Eq>(i: T, j: T) {
        assert i == j;
    }

    fn g<T: copy Eq>(i: T, j: T) {
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
