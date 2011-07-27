


// This is a testcase for issue #94.
fn main() {
    let v: vec[int] = [0, 1, 2, 3, 4, 5];
    let s: str = "abcdef";
    assert (v.(3u) == 3);
    assert (v.(3u8) == 3);
    assert (v.(3i8) == 3);
    assert (v.(3u32) == 3);
    assert (v.(3i32) == 3);
    log v.(3u8);
    assert (s.(3u) == 'd' as u8);
    assert (s.(3u8) == 'd' as u8);
    assert (s.(3i8) == 'd' as u8);
    assert (s.(3u32) == 'd' as u8);
    assert (s.(3i32) == 'd' as u8);
    log s.(3u8);
}