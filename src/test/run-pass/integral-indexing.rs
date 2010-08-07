// This is a testcase for issue #94.

fn main() {

  let vec[int] v = vec(0, 1, 2, 3, 4, 5);
  let str s = "abcdef";
  check (v.(3u) == 3);
  check (v.(3u8) == 3);
  check (v.(3i8) == 3);
  check (v.(3u32) == 3);
  check (v.(3i32) == 3);

  log v.(3u8);

  check (s.(3u) == 'd' as u8);
  check (s.(3u8) == 'd' as u8);
  check (s.(3i8) == 'd' as u8);
  check (s.(3u32) == 'd' as u8);
  check (s.(3i32) == 'd' as u8);

  log s.(3u8);
}