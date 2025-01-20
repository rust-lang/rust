//@ run-pass
//@ edition:2021

// When a closure syntactically captures a place, but doesn't actually capture
// it, make sure MIR building doesn't ICE when handling that place.
//
// Under the Rust 2021 disjoint capture rules, this sort of non-capture can
// occur when a place is only inspected by infallible non-binding patterns.

struct Props {
    field_1: u32, //~ WARNING: fields `field_1` and `field_2` are never read
    field_2: u32,
}

fn main() {
    // Test 1
    let props_2 = Props { field_1: 1, field_2: 1 };

    let _ = || {
        let _: Props = props_2;
    };

    // Test 2
    let mut arr = [1, 3, 4, 5];

    let mref = &mut arr;

    // These array patterns don't need to inspect the array, so the array
    // isn't captured.
    let _c = || match arr {
        [_, _, _, _] => println!("C"),
    };
    let _d = || match arr {
        [_, .., _] => println!("D"),
    };
    let _e = || match arr {
        [_, ..] => println!("E"),
    };

    println!("{:#?}", mref);
}
