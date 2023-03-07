fn main() {
    a();
    b();
    c();
    d();
}

fn a() {
    #[allow(dead_code)]
    union U {
        f1: u32,
        f2: f32,
    }
    let mut u = U { f1: 1 };
    unsafe {
        let b1 = &mut u.f1;
        *b1 = 5;
    }
    assert_eq!(unsafe { u.f1 }, 5);
}

fn b() {
    #[derive(Copy, Clone)]
    struct S {
        x: u32,
        y: u32,
    }

    #[allow(dead_code)]
    union U {
        s: S,
        both: u64,
    }
    let mut u = U { s: S { x: 1, y: 2 } };
    unsafe {
        let bx = &mut u.s.x;
        let by = &mut u.s.y;
        *bx = 5;
        *by = 10;
    }
    assert_eq!(unsafe { u.s.x }, 5);
    assert_eq!(unsafe { u.s.y }, 10);
}

fn c() {
    #[repr(u32)]
    enum Tag {
        I,
        F,
    }

    #[repr(C)]
    union U {
        i: i32,
        f: f32,
    }

    #[repr(C)]
    struct Value {
        tag: Tag,
        u: U,
    }

    fn is_zero(v: Value) -> bool {
        unsafe {
            match v {
                Value { tag: Tag::I, u: U { i: 0 } } => true,
                Value { tag: Tag::F, u: U { f } } => f == 0.0,
                _ => false,
            }
        }
    }
    assert!(is_zero(Value { tag: Tag::I, u: U { i: 0 } }));
    assert!(is_zero(Value { tag: Tag::F, u: U { f: 0.0 } }));
    assert!(!is_zero(Value { tag: Tag::I, u: U { i: 1 } }));
    assert!(!is_zero(Value { tag: Tag::F, u: U { f: 42.0 } }));
}

fn d() {
    union MyUnion {
        f1: u32,
        f2: f32,
    }
    let u = MyUnion { f1: 10 };
    unsafe {
        match u {
            MyUnion { f1: 10 } => {}
            MyUnion { f2: _f2 } => panic!("foo"),
        }
    }
}
