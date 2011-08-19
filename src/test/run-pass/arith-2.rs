

fn main() {
    let i32_c: int = 0x10101010;
    assert (i32_c + i32_c * 2 / 3 * 2 + (i32_c - 7 % 3) ==
                i32_c + i32_c * 2 / 3 * 2 + (i32_c - 7 % 3));
}
