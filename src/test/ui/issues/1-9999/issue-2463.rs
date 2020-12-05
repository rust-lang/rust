// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

struct Pair { f: isize, g: isize }

pub fn main() {

    let x = Pair {
        f: 0,
        g: 0,
    };

    let _y = Pair {
        f: 1,
        g: 1,
        .. x
    };

    let _z = Pair {
        f: 1,
        .. x
    };

}
