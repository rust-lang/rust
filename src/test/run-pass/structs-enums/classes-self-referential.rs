// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


// pretty-expanded FIXME #23616

struct kitten {
    cat: Option<cat>,
}

fn kitten(cat: Option<cat>) -> kitten {
    kitten {
        cat: cat
    }
}

type cat = Box<kitten>;

pub fn main() {}
