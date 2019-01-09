// run-pass

union U {
    a: u64,
    b: u64,
}

const C: U = U { b: 10 };

fn main() {
    unsafe {
        let a = C.a;
        let b = C.b;
        assert_eq!(a, 10);
        assert_eq!(b, 10);
     }
}
