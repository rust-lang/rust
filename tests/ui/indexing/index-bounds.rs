//@ build-fail

fn main() {
    let _n = [64][200];
    //~^ ERROR this operation will panic at runtime [unconditional_panic]

    // issue #121126, test index value between 0xFFFF_FF00 and u32::MAX
    let _n = [64][u32::MAX as usize - 1];
    //~^ ERROR this operation will panic at runtime [unconditional_panic]
}
