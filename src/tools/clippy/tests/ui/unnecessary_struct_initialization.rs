#![allow(clippy::non_canonical_clone_impl, unused)]
#![warn(clippy::unnecessary_struct_initialization)]

struct S {
    f: String,
}

#[derive(Clone, Copy)]
struct T {
    f: u32,
}

struct U {
    f: u32,
}

impl Clone for U {
    fn clone(&self) -> Self {
        // Do not lint: `Self` does not implement `Copy`
        Self { ..*self }
    }
}

#[derive(Copy)]
struct V {
    f: u32,
}

struct W {
    f1: u32,
    f2: u32,
}

impl Clone for V {
    fn clone(&self) -> Self {
        // Lint: `Self` implements `Copy`
        Self { ..*self }
        //~^ unnecessary_struct_initialization
    }
}

fn main() {
    // Should lint: `a` would be consumed anyway
    let a = S { f: String::from("foo") };
    let mut b = S { ..a };
    //~^ unnecessary_struct_initialization

    // Should lint: `b` would be consumed, and is mutable
    let c = &mut S { ..b };
    //~^ unnecessary_struct_initialization

    // Should not lint as `d` is not mutable
    let d = S { f: String::from("foo") };
    let e = &mut S { ..d };

    // Should lint as `f` would be consumed anyway
    let f = S { f: String::from("foo") };
    let g = &S { ..f };
    //~^ unnecessary_struct_initialization

    // Should lint: the result of an expression is mutable
    let h = &mut S {
        //~^ unnecessary_struct_initialization
        ..*Box::new(S { f: String::from("foo") })
    };

    // Should not lint: `m` would be both alive and borrowed
    let m = T { f: 17 };
    let n = &T { ..m };

    // Should not lint: `m` should not be modified
    let o = &mut T { ..m };
    o.f = 32;
    assert_eq!(m.f, 17);

    // Should not lint: `m` should not be modified
    let o = &mut T { ..m } as *mut T;
    unsafe { &mut *o }.f = 32;
    assert_eq!(m.f, 17);

    // Should lint: the result of an expression is mutable and temporary
    let p = &mut T {
        //~^ unnecessary_struct_initialization
        ..*Box::new(T { f: 5 })
    };

    // Should lint: all fields of `q` would be consumed anyway
    let q = W { f1: 42, f2: 1337 };
    let r = W { f1: q.f1, f2: q.f2 };
    //~^ unnecessary_struct_initialization

    // Should not lint: not all fields of `t` from same source
    let s = W { f1: 1337, f2: 42 };
    let t = W { f1: s.f1, f2: r.f2 };

    // Should not lint: different fields of `s` assigned
    let u = W { f1: s.f2, f2: s.f1 };

    // Should lint: all fields of `v` would be consumed anyway
    let v = W { f1: 42, f2: 1337 };
    let w = W { f1: v.f1, ..v };
    //~^ unnecessary_struct_initialization

    // Should not lint: source differs between fields and base
    let x = W { f1: 42, f2: 1337 };
    let y = W { f1: w.f1, ..x };

    // Should lint: range desugars to struct
    let r1 = 0..5;
    let r2 = r1.start..r1.end;
    //~^ unnecessary_struct_initialization

    references();
    shorthand();
}

fn references() {
    // Should not lint as `a` is not mutable
    let a = W { f1: 42, f2: 1337 };
    let b = &mut W { f1: a.f1, f2: a.f2 };

    // Should lint as `d` is a shared reference
    let c = W { f1: 42, f2: 1337 };
    let d = &W { f1: c.f1, f2: c.f2 };
    //~^ unnecessary_struct_initialization

    // Should not lint as `e` is not mutable
    let e = W { f1: 42, f2: 1337 };
    let f = &mut W { f1: e.f1, ..e };

    // Should lint as `h` is a shared reference
    let g = W { f1: 42, f2: 1337 };
    let h = &W { f1: g.f1, ..g };
    //~^ unnecessary_struct_initialization

    // Should not lint as `j` is copy
    let i = V { f: 0x1701d };
    let j = &V { ..i };

    // Should not lint as `k` is copy
    let k = V { f: 0x1701d };
    let l = &V { f: k.f };
}

fn shorthand() {
    struct S1 {
        a: i32,
        b: i32,
    }

    let a = 42;
    let s = S1 { a: 3, b: 4 };

    // Should not lint: `a` is not from `s`
    let s = S1 { a, b: s.b };
}
