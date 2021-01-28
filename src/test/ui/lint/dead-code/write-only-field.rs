#![deny(dead_code)]

struct S {
    f: i32, //~ ERROR: field is never read
    sub: Sub, //~ ERROR: field is never read
}

struct Sub {
    f: i32, //~ ERROR: field is never read
}

fn field_write(s: &mut S) {
    s.f = 1;
    s.sub.f = 2;
}

fn main() {
    let mut s = S { f: 0, sub: Sub { f: 0 } };
    field_write(&mut s);
}
