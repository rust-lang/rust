//@no-rustfix
#![warn(clippy::never_loop)]
#![expect(clippy::needless_return)]

fn main() {
    // diverging closure with no `return`: should trigger
    [0, 1].into_iter().for_each(|x| {
        //~^ never_loop

        let _ = x;
        panic!("boom");
    });

    // benign closure: should NOT trigger
    [0, 1].into_iter().for_each(|x| {
        let _ = x + 1;
    });

    // `return` should NOT trigger even though it is diverging
    [0, 1].into_iter().for_each(|x| {
        println!("x = {x}");
        return;
    });
}
