// Test that move closures drop derefs with `capture_disjoint_fields` enabled.

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
#![feature(rustc_attrs)]

// Test we truncate derefs properly
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
        //~^ NOTE: Capturing ref_s[Deref] -> ByValue
        //~| NOTE: Min Capture ref_s[] -> ByValue
    };
    c();
}

// Test we truncate derefs properly
fn struct_contains_ref_to_another_struct() {
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
        //~^ NOTE: Capturing t[(0, 0),Deref,(0, 0)] -> ByValue
        //~| NOTE: Min Capture t[(0, 0)] -> ByValue
    };

    c();
}

// Test that we don't reduce precision when there is nothing deref.
fn no_ref() {
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

fn main() {
    simple_ref();
    struct_contains_ref_to_another_struct();
    no_ref();
}
