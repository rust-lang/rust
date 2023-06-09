struct S {
    a: u8,
}

union U {
    a: u8,
}

fn main() {
    unsafe {
        let mut s: S;
        let mut u: U;
        s.a = 0; //~ ERROR E0381
        u.a = 0; //~ ERROR E0381
        let sa = s.a;
        let ua = u.a;
    }
}
