// error-pattern: instantiating a sendable type parameter with a copyable type

fn f<T: send>(i: T) {
}

fn main() {
    let i = ~@100;
    f(i);
}