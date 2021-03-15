// Test that we handle derferences properly when only some of the captures are being moved with
// `capture_disjoint_fields` enabled.


#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
#![feature(rustc_attrs)]

#[derive(Debug, Default)]
struct SomeLargeType;
struct MuchLargerType([SomeLargeType; 32]);

// Ensure that we don't capture any derefs when moving captures into the closures,
// i.e. only data from the enclosing stack is moved.
fn big_box() {
    let s = MuchLargerType(Default::default());
    let b = Box::new(s);
    let t = (b, 10);

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ First Pass analysis includes:
    //~| Min Capture analysis includes:
        let p = t.0.0;
        //~^ NOTE: Capturing t[(0, 0),Deref,(0, 0)] -> ImmBorrow
        //~| NOTE: Capturing t[(0, 0)] -> ByValue
        //~| NOTE: Min Capture t[(0, 0)] -> ByValue
        println!("{} {:?}", t.1, p);
        //~^ NOTE: Capturing t[(1, 0)] -> ImmBorrow
        //~| NOTE: Min Capture t[(1, 0)] -> ImmBorrow
    };

    c();
}

fn main() {
    big_box();
}
