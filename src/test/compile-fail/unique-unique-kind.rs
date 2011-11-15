// error-pattern: needed unique type
// xfail-test

fn f<uniq T>(i: T) {
}

fn main() {
    let i = ~@100;
    f(i);
}