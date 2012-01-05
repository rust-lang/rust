fn sendable() {

    fn f<T: send>(i: T, j: T) {
        assert i == j;
    }

    fn g<T: send>(i: T, j: T) {
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

    fn f<T: copy>(i: T, j: T) {
        assert i == j;
    }

    fn g<T: copy>(i: T, j: T) {
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

    fn f<T>(i: T, j: T) {
        assert i == j;
    }

    fn g<T>(i: T, j: T) {
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
