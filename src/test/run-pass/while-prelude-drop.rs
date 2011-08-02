
tag t { a; b(str); }

fn make(i: int) -> t {
    if i > 10 { ret a; }
    let s = "hello";
    // Ensure s is non-const.

    s += "there";
    ret b(s);
}

fn main() {
    let i = 0;


    // The auto slot for the result of make(i) should not leak.
    while make(i) != a { i += 1; }
}