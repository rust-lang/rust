fn f<T: Send>(_i: T) {
}

fn main() {
    let i = ~@100;
    f(i); //~ ERROR missing `send`
}
