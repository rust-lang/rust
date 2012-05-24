fn f<T: send>(_i: T) {
}

fn main() {
    let i = ~@100;
    f(i); //! ERROR missing `send`
}
