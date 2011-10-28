fn unique() {

    fn f<uniq T>(i: T, j: T) {
        assert i == j;
    }

    fn g<uniq T>(i: T, j: T) {
        assert i != j;
    }

    let i = ~100;
    let j = ~100;
    f(i, j);
    let i = ~100;
    let j = ~101;
    g(i, j);
}

fn shared() {

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

fn pinned() {

    fn f<pin T>(i: T, j: T) {
        assert i == j;
    }

    fn g<pin T>(i: T, j: T) {
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
    unique();
    shared();
    pinned();
}