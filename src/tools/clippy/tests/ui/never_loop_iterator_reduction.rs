//@no-rustfix
#![warn(clippy::never_loop)]

fn main() {
    // diverging closure: should trigger
    [0, 1].into_iter().for_each(|x| {
        //~^ never_loop

        let _ = x;
        panic!("boom");
    });

    // benign closure: should NOT trigger
    [0, 1].into_iter().for_each(|x| {
        let _ = x + 1;
    });
}
