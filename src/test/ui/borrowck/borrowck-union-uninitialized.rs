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
        s.a = 0; //~ ERROR assign to part of possibly uninitialized variable: `s`
        u.a = 0; //~ ERROR assign to part of possibly uninitialized variable: `u`
        let sa = s.a;
        let ua = u.a;
    }
}
