//@ edition:2021

enum SingleVariant {
    A
}

struct TestStruct {
    x: i32,
    y: i32,
    z: i32,
}

fn edge_case_if() {
    let sv = SingleVariant::A;
    let condition = true;
    // sv should be captured, regardless of being SingleVariant
    let _a = || {
        match sv {
            SingleVariant::A if condition => (),
            _ => ()
        }
    };
    let mut mut_sv = sv;
    //~^ ERROR: cannot move out of `sv` because it is borrowed
    _a();

    // ts should be captured
    let ts = TestStruct { x: 1, y: 1, z: 1 };
    let _b = || { match ts {
        TestStruct{ x: 1, .. } => (),
        _ => ()
    }};
    let mut mut_ts = ts;
    //~^ ERROR: cannot move out of `ts` because it is borrowed
    _b();
}

struct Unit;

enum TSingle {
    A(u32, u32),
}

enum SSingle {
    A { a: u32, b: u32 },
}

struct TStruct(u32, u32);
struct SStruct { a: u32, b: u32 }


// Matching on a unit struct should not capture it
fn match_unit_struct(mut x: (Unit, u32)) {
    let r = &mut x.0;
    let _ = || {
        let (Unit, a) = x;
        a
    };

    let _ = *r;
}

// However, in the equivalent test case with a single-variant unit enum,
// the unit enum *should* be captured.
fn match_unit_enum(mut x: (SingleVariant, u32)) {
    let r = &mut x.0;
    let _ = || {
    //~^ ERROR: cannot borrow `x.0` as immutable
        let (SingleVariant::A, a) = x;
        a
    };

    let _ = *r;
}

// Matching on a struct should only capture the fields being touched
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

// However, matching on a single-variant enum with fields inside
// does not behave like a struct: the entire enum is captured, as
// the (zero-size) discriminant must be read.
fn match_singleton(mut x: SSingle) {
    let SSingle::A { a: ref mut r, .. } = x;
    let _ = || {
    //~^ ERROR: cannot borrow `x` as immutable
        let SSingle::A { b, .. } = x;
        b
    };

    let _ = *r;
}

fn match_tuple_singleton(mut x: TSingle) {
    let TSingle::A(ref mut r, _) = x;
    let _ = || {
    //~^ ERROR: cannot borrow `x` as immutable
        let TSingle::A(_, a) = x;
        a
    };

    let _ = *r;
}

fn main() {}
