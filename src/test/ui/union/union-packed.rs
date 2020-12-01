// run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

use std::mem::{size_of, size_of_val, align_of, align_of_val};

struct S {
    a: u16,
    b: [u8; 3],
}

#[repr(packed)]
struct Sp1 {
    a: u16,
    b: [u8; 3],
}

#[repr(packed(2))]
struct Sp2 {
    a: u16,
    b: [u8; 3],
}

union U {
    a: u16,
    b: [u8; 3],
}

#[repr(packed)]
union Up1 {
    a: u16,
    b: [u8; 3],
}

#[repr(packed(2))]
union Up2 {
    a: u16,
    b: [u8; 3],
}

#[repr(C, packed(4))]
union Up4c {
    a: u16,
    b: [u8; 3],
}

const CS: S = S { a: 0, b: [0, 0, 0] };
const CSP1: Sp1 = Sp1 { a: 0, b: [0, 0, 0] };
const CSP2: Sp2 = Sp2 { a: 0, b: [0, 0, 0] };
const CU: U = U { b: [0, 0, 0] };
const CUP1: Up1 = Up1 { b: [0, 0, 0] };
const CUP2: Up2 = Up2 { b: [0, 0, 0] };
const CUP4C: Up4c = Up4c { b: [0, 0, 0] };

fn main() {
    let s = S { a: 0, b: [0, 0, 0] };
    assert_eq!(size_of::<S>(), 6);
    assert_eq!(size_of_val(&s), 6);
    assert_eq!(size_of_val(&CS), 6);
    assert_eq!(align_of::<S>(), 2);
    assert_eq!(align_of_val(&s), 2);
    assert_eq!(align_of_val(&CS), 2);

    let sp1 = Sp1 { a: 0, b: [0, 0, 0] };
    assert_eq!(size_of::<Sp1>(), 5);
    assert_eq!(size_of_val(&sp1), 5);
    assert_eq!(size_of_val(&CSP1), 5);
    assert_eq!(align_of::<Sp1>(), 1);
    assert_eq!(align_of_val(&sp1), 1);
    assert_eq!(align_of_val(&CSP1), 1);

    let sp2 = Sp2 { a: 0, b: [0, 0, 0] };
    assert_eq!(size_of::<Sp2>(), 6);
    assert_eq!(size_of_val(&sp2), 6);
    assert_eq!(size_of_val(&CSP2), 6);
    assert_eq!(align_of::<Sp2>(), 2);
    assert_eq!(align_of_val(&sp2), 2);
    assert_eq!(align_of_val(&CSP2), 2);

    let u = U { b: [0, 0, 0] };
    assert_eq!(size_of::<U>(), 4);
    assert_eq!(size_of_val(&u), 4);
    assert_eq!(size_of_val(&CU), 4);
    assert_eq!(align_of::<U>(), 2);
    assert_eq!(align_of_val(&u), 2);
    assert_eq!(align_of_val(&CU), 2);

    let Up1 = Up1 { b: [0, 0, 0] };
    assert_eq!(size_of::<Up1>(), 3);
    assert_eq!(size_of_val(&Up1), 3);
    assert_eq!(size_of_val(&CUP1), 3);
    assert_eq!(align_of::<Up1>(), 1);
    assert_eq!(align_of_val(&Up1), 1);
    assert_eq!(align_of_val(&CUP1), 1);

    let up2 = Up2 { b: [0, 0, 0] };
    assert_eq!(size_of::<Up2>(), 4);
    assert_eq!(size_of_val(&up2), 4);
    assert_eq!(size_of_val(&CUP2), 4);
    assert_eq!(align_of::<Up2>(), 2);
    assert_eq!(align_of_val(&up2), 2);
    assert_eq!(align_of_val(&CUP2), 2);

    let up4c = Up4c { b: [0, 0, 0] };
    assert_eq!(size_of::<Up4c>(), 4);
    assert_eq!(size_of_val(&up4c), 4);
    assert_eq!(size_of_val(&CUP4C), 4);
    assert_eq!(align_of::<Up4c>(), 2);
    assert_eq!(align_of_val(&up4c), 2);
    assert_eq!(align_of_val(&CUP4C), 2);

    hybrid::check_hybrid();
}

mod hybrid {
    use std::mem::{size_of, align_of};

    #[repr(packed)]
    #[derive(Copy, Clone)]
    struct S1 {
        a: u16,
        b: u8,
    }

    #[repr(packed)]
    union U {
        s: S1,
        c: u16,
    }

    #[repr(packed)]
    struct S2 {
        d: u8,
        u: U,
    }

    #[repr(C, packed(2))]
    struct S1C {
        a: u16,
        b: u8,
    }

    #[repr(C, packed(2))]
    union UC {
        s: S1,
        c: u16,
    }

    #[repr(C, packed(2))]
    struct S2C {
        d: u8,
        u: UC,
    }

    pub fn check_hybrid() {
        assert_eq!(align_of::<S1>(), 1);
        assert_eq!(size_of::<S1>(), 3);
        assert_eq!(align_of::<U>(), 1);
        assert_eq!(size_of::<U>(), 3);
        assert_eq!(align_of::<S2>(), 1);
        assert_eq!(size_of::<S2>(), 4);

        assert_eq!(align_of::<S1C>(), 2);
        assert_eq!(size_of::<S1C>(), 4);
        assert_eq!(align_of::<UC>(), 2);
        assert_eq!(size_of::<UC>(), 4);
        assert_eq!(align_of::<S2C>(), 2);
        assert_eq!(size_of::<S2C>(), 6);
    }
}
