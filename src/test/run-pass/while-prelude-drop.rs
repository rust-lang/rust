
enum t { a, b(~str), }

fn make(i: int) -> t {
    if i > 10 { return a; }
    let mut s = ~"hello";
    // Ensure s is non-const.

    s += ~"there";
    return b(s);
}

fn main() {
    let mut i = 0;


    // The auto slot for the result of make(i) should not leak.
    while make(i) != a { i += 1; }
}
