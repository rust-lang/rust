// run-pass
#![allow(non_camel_case_types)]

enum blah { a, b, }

fn or_alt(q: blah) -> isize {
  match q { blah::a | blah::b => { 42 } }
}

pub fn main() {
    assert_eq!(or_alt(blah::a), 42);
    assert_eq!(or_alt(blah::b), 42);
}
