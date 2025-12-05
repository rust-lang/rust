//@check-pass

fn main() {
    let _x = -1_i32 >> -1;
    #[expect(overflowing_literals)]
    let _y = 1u32 >> 10000000000000u32;
}
