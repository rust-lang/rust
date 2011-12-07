fn main() {
    assert 0xffffffffu32 == (-1 as u32);
    assert 4294967295u32 == (-1 as u32);
    assert 0xffffffffffffffffu64 == (-1 as u64);
    assert 18446744073709551615u64 == (-1 as u64);

    assert -2147483648i32 - 1i32 == 2147483647i32;
    assert -9223372036854775808i64 - 1i64 == 9223372036854775807i64;
}