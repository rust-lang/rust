// run-pass

// Test that ByValue captures compile sucessefully especially when the captures are
// derefenced within the closure.

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

#[derive(Debug, Default)]
struct SomeLargeType;
struct MuchLargerType([SomeLargeType; 32]);

fn big_box() {
    let s = MuchLargerType(Default::default());
    let b = Box::new(s);
    let t = (b, 10);

    let c = || {
        let p = t.0.0;
        println!("{} {:?}", t.1, p);
    };

    c();
}

fn main() {
    big_box();
}
