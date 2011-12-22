fn even(&&e: int) -> bool {
    e % 2 == 0
}

fn log_if<T>(c: fn(T)->bool, e: T) {
    if c(e) { log_full(core::debug, e); }
}

fn main() {
    (bind log_if(even, _))(2);
}
