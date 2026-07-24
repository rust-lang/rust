//@ edition: 2024
//@ compile-flags: -Znext-solver -Zassumptions-on-binders -Zdxf

#![allow(dead_code)]
use std::marker::PhantomData;

struct Guarded<'a, 'b> {
    _p: PhantomData<(fn(&'a ()) -> &'a (), fn(&'b ()) -> &'b ())>,
}
unsafe impl<'a, 'b: 'a> Send for Guarded<'a, 'b> {}

async fn yield_point() {}

async fn use_guarded<'a, 'b>(data1: &'a u8, data2: &'b u8) {
    let mut r1 = data1; // creates local '?r1, adds NLL constraint 'a: '?r1
    let r2 = data2; // creates local '?r2, adds NLL constraint 'b: '?r2
    r1 = r2; // assignment adds NLL constraint '?r2: '?r1, so 'b: '?r1
    // Note: 'a: '?r1 and 'b: '?r1 simply means '?r1 is bounded by the intersection of 'a and 'b.
    // It does NOT prove 'b: 'a or 'a: 'b in the generic environment of use_guarded.
    let g = Guarded::<'a, 'b> { _p: PhantomData };
    yield_point().await;
    drop(g);
}

fn assert_send(_: impl Send) {}

fn main() {
    let data1 = 1u8; // outer scope ('data1)
    {
        let data2 = 2u8; // inner scope ('data2)
        assert_send(use_guarded(&data1, &data2));
        //~^ ERROR cannot be sent between threads safely
        // In caller scope, 'data1 outlives 'data2, but 'data2 does NOT outlive 'data1.
        // Since use_guarded does not internally prove 'b: 'a, Guarded<'a, 'b> cannot implement Send.
        // The plan is to use NLL to guide the diagnostics.
    }
}
