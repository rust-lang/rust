#![feature(capture_disjoint_fields)]
#![feature(rustc_attrs)]

struct S {
    a: String,
    b: String,
}

fn main() {
    let t = (String::new(), String::new());

    let s = S {
        a: String::new(),
        b: String::new(),
    };

    let c = #[rustc_capture_analysis] || {
        let (t1, t2) = t;
    };


    // MIR Build
    //
    // Create place for the initalizer in let which is `t`
    //
    // I'm reading Field 1 from `t`, so apply Field projections;
    //
    // new place -> t[1]
    //
    // I'm reading Field 2 from `t`, so apply Field projections;
    //
    // new place -> t[2]
    //
    // New
    // ---------
    //
    // I'm building something starting at `t`
    //
    // I read field 1 from `t`
    //
    // I need to use `t[1]`, therefore the place must be constructable
    //
    // Find the capture index for `t[1]` for this closure.
    //
    // I read field 2 from `t`
    //
    // I need to use `t[2]`, therefore the place must be constructable
    //
    // Find the capture index for `t[2]` for this closure.

    c();
}
