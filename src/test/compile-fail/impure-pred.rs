// -*- rust -*-

fn g() { }

pure fn f(_q: int) -> bool {
    g(); //! ERROR access to impure function prohibited in pure context
    ret true;
}

fn main() {
}
