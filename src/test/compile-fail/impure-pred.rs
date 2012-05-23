// -*- rust -*-

fn g() { }

pure fn f(_q: int) -> bool {
    g(); //! ERROR access to non-pure functions prohibited in pure context
    ret true;
}

fn main() {
    let x = 0;

    check (f(x));
}
