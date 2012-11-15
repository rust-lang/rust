
enum t { a, b(~str), }

impl t : cmp::Eq {
    pure fn eq(&self, other: &t) -> bool {
        match *self {
            a => {
                match (*other) {
                    a => true,
                    b(_) => false
                }
            }
            b(s0) => {
                match (*other) {
                    a => false,
                    b(s1) => s0 == s1
                }
            }
        }
    }
    pure fn ne(&self, other: &t) -> bool { !(*self).eq(other) }
}

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
