fn sendable() {

    fn f<send T>(i: T, j: T) {
        assert i == j;
    }

    fn g<send T>(i: T, j: T) {
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

    fn f<copy T>(i: T, j: T) {
        assert i == j;
    }

    fn g<copy T>(i: T, j: T) {
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
