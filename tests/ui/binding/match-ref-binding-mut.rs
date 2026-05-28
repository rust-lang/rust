//@ run-pass
#![allow(non_shorthand_field_patterns)]

struct Rec {
    f: isize
}

fn destructure(x: &mut Rec) {
    match *x {
      Rec {f: ref mut f} => *f += 1
    }
}

pub fn main() {
    let mut v = Rec {f: 22};
    destructure(&mut v);
    assert_eq!(v.f, 23);
}
