// Test that move closures drop derefs with `capture_disjoint_fields` enabled.

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
#![feature(rustc_attrs)]

fn simple_move_closure() {
    struct S(String);
    struct T(S);

    let t = T(S("s".into()));
    let mut c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    move || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        t.0.0 = "new S".into();
        //~^ NOTE: Capturing t[(0, 0),(0, 0)] -> ByValue
        //~| NOTE: Min Capture t[(0, 0),(0, 0)] -> ByValue
    };
    c();
}

// Test move closure use reborrows when using references
fn simple_ref() {
    let mut s = 10;
    let ref_s = &mut s;

    let mut c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    move || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        *ref_s += 10;
        //~^ NOTE: Capturing ref_s[Deref] -> UniqueImmBorrow
        //~| NOTE: Min Capture ref_s[Deref] -> UniqueImmBorrow
    };
    c();
}

// Test move closure use reborrows when using references
fn struct_contains_ref_to_another_struct_1() {
    struct S(String);
    struct T<'a>(&'a mut S);

    let mut s = S("s".into());
    let t = T(&mut s);

    let mut c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    move || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        t.0.0 = "new s".into();
        //~^ NOTE: Capturing t[(0, 0),Deref,(0, 0)] -> UniqueImmBorrow
        //~| NOTE: Min Capture t[(0, 0),Deref,(0, 0)] -> UniqueImmBorrow
    };

    c();
}

// Test that we can use reborrows to read data of Copy types
// i.e. without truncating derefs
fn struct_contains_ref_to_another_struct_2() {
    struct S(i32);
    struct T<'a>(&'a S);

    let s = S(0);
    let t = T(&s);

    let mut c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    move || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        let _t = t.0.0;
        //~^ NOTE: Capturing t[(0, 0),Deref,(0, 0)] -> ImmBorrow
        //~| NOTE: Min Capture t[(0, 0),Deref,(0, 0)] -> ImmBorrow
    };

    c();
}

// Test that we can use truncate to move out of !Copy types
fn struct_contains_ref_to_another_struct_3() {
    struct S(String);
    struct T<'a>(&'a S);

    let s = S("s".into());
    let t = T(&s);

    let mut c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    move || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        let _t = t.0.0;
        //~^ NOTE: Capturing t[(0, 0),Deref,(0, 0)] -> ImmBorrow
        //~| NOTE: Capturing t[(0, 0)] -> ByValue
        //~| NOTE: Min Capture t[(0, 0)] -> ByValue
    };

    c();
}

// Test that derefs of box are truncated in move closures
fn truncate_box_derefs() {
    struct S(i32);

    let b = Box::new(S(10));

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    move || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        let _t = b.0;
        //~^ NOTE: Capturing b[Deref,(0, 0)] -> ByValue
        //~| NOTE: Capturing b[] -> ByValue
        //~| NOTE: Min Capture b[] -> ByValue
    };

    c();
}

fn main() {
    simple_move_closure();
    simple_ref();
    struct_contains_ref_to_another_struct_1();
    struct_contains_ref_to_another_struct_2();
    struct_contains_ref_to_another_struct_3();
    truncate_box_derefs();
}
