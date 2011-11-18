// error-pattern: instantiating a sendable type parameter with a copyable type

fn f<send T>(i: T) {
}

fn main() {
    let i = ~@100;
    f(i);
}