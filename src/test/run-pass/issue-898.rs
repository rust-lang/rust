fn even(&&e: int) -> bool {
    e % 2 == 0
}

fn log_if<T>(c: native fn(T)->bool, e: T) {
    if c(e) { log(debug, e); }
}

fn main() {
    (bind log_if(even, _))(2);
}
