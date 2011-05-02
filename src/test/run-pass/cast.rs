// -*- rust -*-


fn main() {
  let int i = 'Q' as int;
  check (i == 0x51);
  let u32 u = i as u32;
  check (u == (0x51 as u32));
  check (u == ('Q' as u32));
  check ((i as u8) == ('Q' as u8));
  check (((i as u8) as i8) == (('Q' as u8) as i8));
  check ((0x51 as char) == 'Q');

  check (true == (1 as bool));
  check ((0 as u32) == (false as u32));
}
