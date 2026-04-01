#![feature(coverage_attribute)]
//@ edition: 2024
//@ revisions: base auto on

// Tests for how `#[automatically_derived]` affects coverage instrumentation.
//
// The actual behaviour is an implementation detail, so this test mostly exists
// to show when that behaviour has been accidentally or deliberately changed.
//
// Revision guide:
// - base: Test baseline instrumentation behaviour without `#[automatically_derived]`
// - auto: Test how `#[automatically_derived]` affects instrumentation
// - on:   Test interaction between auto-derived and `#[coverage(on)]`

struct MyStruct;

trait MyTrait {
    fn my_assoc_fn();
}

#[cfg_attr(auto, automatically_derived)]
#[cfg_attr(on, automatically_derived)]
#[cfg_attr(on, coverage(on))]
impl MyTrait for MyStruct {
    fn my_assoc_fn() {
        fn inner_fn() {
            say("in inner fn");
        }

        #[coverage(on)]
        fn inner_fn_on() {
            say("in inner fn (on)");
        }

        let closure = || {
            say("in closure");
        };

        closure();
        inner_fn();
        inner_fn_on();
    }
}

#[coverage(off)]
#[inline(never)]
fn say(s: &str) {
    println!("{s}");
}

fn main() {
    MyStruct::my_assoc_fn();
}
