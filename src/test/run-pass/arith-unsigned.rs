


// Unsigned integer operations
fn main() {
    assert (0u8 < 255u8);
    assert (0u8 <= 255u8);
    assert (255u8 > 0u8);
    assert (255u8 >= 0u8);
    assert (250u8 / 10u8 == 25u8);
    assert (255u8 % 10u8 == 5u8);
    assert (0u16 < 60000u16);
    assert (0u16 <= 60000u16);
    assert (60000u16 > 0u16);
    assert (60000u16 >= 0u16);
    assert (60000u16 / 10u16 == 6000u16);
    assert (60005u16 % 10u16 == 5u16);
    assert (0u32 < 4000000000u32);
    assert (0u32 <= 4000000000u32);
    assert (4000000000u32 > 0u32);
    assert (4000000000u32 >= 0u32);
    assert (4000000000u32 / 10u32 == 400000000u32);
    assert (4000000005u32 % 10u32 == 5u32);
    // 64-bit numbers have some flakiness yet. Not tested

}