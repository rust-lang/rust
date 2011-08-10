tag maybe[T] { nothing; just(T); }

fn foo(x: maybe<int>) {
    alt x { nothing. { log_err "A"; } just(a) { log_err "B"; } }
}

fn main() { }
