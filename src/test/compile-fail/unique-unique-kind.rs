fn f<T: Send>(_i: T) {
}

fn main() {
    let i = ~@100;
    f(move i); //~ ERROR missing `send`
}
