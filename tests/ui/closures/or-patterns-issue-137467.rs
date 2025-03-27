//@ edition:2024
//@ check-pass

const X: u32 = 0;

fn match_literal(x: (u32, u32, u32)) {
    let _ = || {
        let ((0, a, _) | (_, _, a)) = x;
        a
    };
}

fn match_range(x: (u32, u32, u32)) {
    let _ = || {
        let ((0..5, a, _) | (_, _, a)) = x;
        a
    };
}

fn match_const(x: (u32, u32, u32)) {
    let _ = || {
        let ((X, a, _) | (_, _, a)) = x;
        a
    };
}

// related testcase reported in #138973
fn without_bindings(x: u32) {
    let _ = || {
        let (0 | _) = x;
    };
}

enum Choice { A, B }

fn match_unit_variant(x: (Choice, u32, u32)) {
    let _ = || {
        let ((Choice::A, a, _) | (Choice::B, _, a)) = x;
        a
    };
}

struct Unit;

fn match_unit_struct(mut x: (Unit, u32)) {
    let r = &mut x.0;
    let _ = || {
        let (Unit, a) = x;
        a
    };

    let _ = *r;
}

enum Also { Unit }

fn match_unit_enum(mut x: (Also, u32)) {
    let r = &mut x.0;
    let _ = || {
        let (Also::Unit, a) = x;
        a
    };

    let _ = *r;
}

enum TEnum {
    A(u32),
    B(u32),
}

enum SEnum {
    A { a: u32 },
    B { a: u32 },
}

fn match_tuple_enum(x: TEnum) {
    let _ = || {
        let (TEnum::A(a) | TEnum::B(a)) = x;
        a
    };
}

fn match_struct_enum(x: SEnum) {
    let _ = || {
        let (SEnum::A { a } | SEnum::B { a }) = x;
        a
    };
}

enum TSingle {
    A(u32, u32),
}

enum SSingle {
    A { a: u32, b: u32 },
}

struct TStruct(u32, u32);
struct SStruct { a: u32, b: u32 }

fn match_struct(mut x: SStruct) {
    let r = &mut x.a;
    let _ = || {
        let SStruct { b, .. } = x;
        b
    };

    let _ = *r;
}

fn match_tuple_struct(mut x: TStruct) {
    let r = &mut x.0;
    let _ = || {
        let TStruct(_, a) = x;
        a
    };

    let _ = *r;
}

fn match_singleton(mut x: SSingle) {
    let SSingle::A { a: ref mut r, .. } = x;
    let _ = || {
        let SSingle::A { b, .. } = x;
        b
    };

    let _ = *r;
}

fn match_tuple_singleton(mut x: TSingle) {
    let TSingle::A(ref mut r, _) = x;
    let _ = || {
        let TSingle::A(_, a) = x;
        a
    };

    let _ = *r;
}

fn match_slice(x: (&[u32], u32, u32)) {
    let _ = || {
        let (([], a, _) | ([_, ..], _, a)) = x;
        a
    };
}

// Original testcase, for completeness
enum Camera {
    Normal { base_transform: i32 },
    Volume { transform: i32 },
}

fn draw_ui(camera: &mut Camera) {
    || {
        let (Camera::Normal {
            base_transform: _transform,
        }
        | Camera::Volume {
            transform: _transform,
        }) = camera;
    };
}

fn draw_ui2(camera: &mut Camera) {
    || {
        let (Camera::Normal {
            base_transform: _,
        }
        | Camera::Volume {
            transform: _,
        }) = camera;
    };
}

fn main() {}
