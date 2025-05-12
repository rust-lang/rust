//@ edition:2021

#![feature(rustc_attrs)]

// `u8` aligned at a byte and are unaffected by repr(packed).
// Therefore we *could* precisely (and safely) capture references to both the fields,
// but we don't, since we don't want capturing to change when field types change alignment.
fn test_alignment_not_affected() {
    #[repr(packed)]
    struct Foo { x: u8, y: u8 }

    let mut foo = Foo { x: 0, y: 0 };

    let mut c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        let z1: &u8 = &foo.x;
        //~^ NOTE: Capturing foo[] -> Immutable
        let z2: &mut u8 = &mut foo.y;
        //~^ NOTE: Capturing foo[] -> Mutable
        //~| NOTE: Min Capture foo[] -> Mutable

        *z2 = 42;

        println!("({}, {})", z1, z2);
    };

    c();
}

// `String`, `u16` are not aligned at a one byte boundary and are thus affected by repr(packed).
//
// Here we test that the closure doesn't capture a reference point to `foo.x` but
// rather capture `foo` entirely.
fn test_alignment_affected() {
    #[repr(packed)]
    struct Foo { x: String, y: u16 }

    let mut foo = Foo { x: String::new(), y: 0 };

    let mut c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        let z1: &String = &foo.x;
        //~^ NOTE: Capturing foo[] -> Immutable
        let z2: &mut u16 = &mut foo.y;
        //~^ NOTE: Capturing foo[] -> Mutable
        //~| NOTE: Min Capture foo[] -> Mutable


        *z2 = 42;

        println!("({}, {})", z1, z2);
    };

    c();
}

// Given how the closure desugaring is implemented (at least at the time of writing this test),
// we don't need to truncate the captured path to a reference into a packed-struct if the field
// being referenced will be moved into the closure, since it's safe to move out a field from a
// packed-struct.
//
// However to avoid surprises for the user, or issues when the closure is
// inlined we will truncate the capture to access just the struct regardless of if the field
// might get moved into the closure.
fn test_truncation_when_ref_and_move() {
    #[repr(packed)]
    struct Foo { x: String }

    let mut foo = Foo { x: String::new() };

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{}", foo.x);
        //~^ NOTE: Capturing foo[] -> Immutable
        //~| NOTE: Min Capture foo[] -> ByValue
        //~| NOTE: foo[] used here
        let _z = foo.x;
        //~^ NOTE: Capturing foo[(0, 0)] -> ByValue
        //~| NOTE: foo[] captured as ByValue here
    };

    c();
}

fn main() {
    test_truncation_when_ref_and_move();
    test_alignment_affected();
    test_alignment_not_affected();
}
