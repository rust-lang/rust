// run-pass
// Issue 23030: Workaround overflowing discriminant
// with explicit assignments.

// See also compile-fail/overflow-discrim.rs, which shows what
// happens if you leave the OhNo explicit cases out here.

fn f_i8() {
    #[repr(i8)]
    enum A {
        Ok = i8::MAX - 1,
        Ok2,
        OhNo = i8::MIN,
        NotTheEnd = -1,
        Zero,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
    let z = (A::NotTheEnd, A::Zero).1 as i8;
    assert_eq!(z, 0);
}

fn f_u8() {
    #[repr(u8)]
    enum A {
        Ok = u8::MAX - 1,
        Ok2,
        OhNo = u8::MIN,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
}

fn f_i16() {
    #[repr(i16)]
    enum A {
        Ok = i16::MAX - 1,
        Ok2,
        OhNo = i16::MIN,
        NotTheEnd = -1,
        Zero,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
    let z = (A::NotTheEnd, A::Zero).1 as i16;
    assert_eq!(z, 0);
}

fn f_u16() {
    #[repr(u16)]
    enum A {
        Ok = u16::MAX - 1,
        Ok2,
        OhNo = u16::MIN,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
}

fn f_i32() {
    #[repr(i32)]
    enum A {
        Ok = i32::MAX - 1,
        Ok2,
        OhNo = i32::MIN,
        NotTheEnd = -1,
        Zero,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
    let z = (A::NotTheEnd, A::Zero).1 as i32;
    assert_eq!(z, 0);
}

fn f_u32() {
    #[repr(u32)]
    enum A {
        Ok = u32::MAX - 1,
        Ok2,
        OhNo = u32::MIN,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
}

fn f_i64() {
    #[repr(i64)]
    enum A {
        Ok = i64::MAX - 1,
        Ok2,
        OhNo = i64::MIN,
        NotTheEnd = -1,
        Zero,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
    let z = (A::NotTheEnd, A::Zero).1 as i64;
    assert_eq!(z, 0);
}

fn f_u64() {
    #[repr(u64)]
    enum A {
        Ok = u64::MAX - 1,
        Ok2,
        OhNo = u64::MIN,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
}

fn f_isize() {
    #[repr(isize)]
    enum A {
        Ok = isize::MAX - 1,
        Ok2,
        OhNo = isize::MIN,
        NotTheEnd = -1,
        Zero,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
    let z = (A::NotTheEnd, A::Zero).1 as isize;
    assert_eq!(z, 0);
}

fn f_usize() {
    #[repr(usize)]
    enum A {
        Ok = usize::MAX - 1,
        Ok2,
        OhNo = usize::MIN,
    }

    let _x = (A::Ok, A::Ok2, A::OhNo);
}

fn main() {
    f_i8(); f_u8();
    f_i16(); f_u16();
    f_i32(); f_u32();
    f_i64(); f_u64();

    f_isize(); f_usize();
}
