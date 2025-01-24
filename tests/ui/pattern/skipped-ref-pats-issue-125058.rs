//@ run-pass
//@ edition: 2024

#![allow(incomplete_features)]
#![feature(ref_pat_eat_one_layer_2024)]

struct Foo;
//~^ WARN struct `Foo` is never constructed

fn main() {
    || {
        //~^ WARN unused closure that must be used
        if let Some(Some(&mut x)) = &mut Some(&mut Some(0)) {
            let _: u32 = x;
        }
    };
}
