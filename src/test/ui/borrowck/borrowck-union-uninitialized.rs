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
        s.a = 0;
        u.a = 0;
        let sa = s.a; //~ ERROR use of possibly uninitialized variable: `s.a`
        let ua = u.a; //~ ERROR use of possibly uninitialized variable: `u.a`
    }
}
