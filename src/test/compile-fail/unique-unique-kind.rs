// error-pattern: needed unique type

fn f<~T>(i: T) {
}

fn main() {
    let i = ~@100;
    f(i);
}