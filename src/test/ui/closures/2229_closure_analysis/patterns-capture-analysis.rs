#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
#![feature(rustc_attrs)]

// Should capture the discriminant since a variant of a multivariant enum is
// mentioned in the match arm; the discriminant is captured by the closure regardless
// of if it creates a binding
fn test_1_should_capture() {
    let variant = Some(2229);
    let c =  #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>

    || {
    //~^ First Pass analysis includes:
    //~| Min Capture analysis includes:
        match variant {
        //~^ NOTE: Capturing variant[] -> ImmBorrow
        //~| NOTE: Min Capture variant[] -> ImmBorrow
            Some(_) => {}
            _ => {}
        }
    };
    c();
}

// Should not capture the discriminant since only a wildcard is mentioned in the
// match arm
fn test_2_should_not_capture() {
    let variant = Some(2229);
    let c =  #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ First Pass analysis includes:
        match variant {
            _ => {}
        }
    };
    c();
}

// Testing single variant patterns
enum SingleVariant {
    Points(u32)
}

// Should not capture the discriminant since the single variant mentioned
// in the match arm does not trigger a binding
fn test_3_should_not_capture_single_variant() {
    let variant = SingleVariant::Points(1);
    let c =  #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ First Pass analysis includes:
        match variant {
            SingleVariant::Points(_) => {}
        }
    };
    c();
}

// Should not capture the discriminant since the single variant mentioned
// in the match arm does not trigger a binding
fn test_6_should_capture_single_variant() {
    let variant = SingleVariant::Points(1);
    let c =  #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ First Pass analysis includes:
    //~| Min Capture analysis includes:
        match variant {
            //~^ NOTE: Capturing variant[] -> ImmBorrow
            //~| NOTE: Capturing variant[(0, 0)] -> ImmBorrow
            //~| NOTE: Min Capture variant[] -> ImmBorrow
            SingleVariant::Points(a) => {
                println!("{:?}", a);
            }
        }
    };
    c();
}

// Should not capture the discriminant since only wildcards are mentioned in the
// match arm
fn test_4_should_not_capture_array() {
    let array: [i32; 3] = [0; 3];
    let c =  #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ First Pass analysis includes:
        match array {
            [_,_,_] => {}
        }
    };
    c();
}

// Testing MultiVariant patterns
enum MVariant {
    A,
    B,
    C,
}

// Should capture the discriminant since a variant of the multi variant enum is
// mentioned in the match arm; the discriminant is captured by the closure
// regardless of if it creates a binding
fn test_5_should_capture_multi_variant() {
    let variant = MVariant::A;
    let c =  #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ First Pass analysis includes:
    //~| Min Capture analysis includes:
        match variant {
        //~^ NOTE: Capturing variant[] -> ImmBorrow
        //~| NOTE: Min Capture variant[] -> ImmBorrow
            MVariant::A => {}
            _ => {}
        }
    };
    c();
}

fn main() {
    test_1_should_capture();
    test_2_should_not_capture();
    test_3_should_not_capture_single_variant();
    test_6_should_capture_single_variant();
    test_4_should_not_capture_array();
    test_5_should_capture_multi_variant();
}
