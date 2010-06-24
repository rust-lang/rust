
import size_of = std.sys.rustrt.size_of;

use std;

fn main() {
  check (size_of[u8]() == uint(1));
  check (size_of[u32]() == uint(4));
  check (size_of[char]() == uint(4));
  check (size_of[i8]() == uint(1));
  check (size_of[i32]() == uint(4));
  check (size_of[tup(u8,i8)]() == uint(2));
  check (size_of[tup(u8,i8,u8)]() == uint(3));
  // Alignment causes padding before the char and the u32.
  check (size_of[tup(u8,i8,tup(char,u8),u32)]() == uint(16));
  check (size_of[int]() == size_of[uint]());
  check (size_of[tup(int,())]() == size_of[int]());
  check (size_of[tup(int,(),())]() == size_of[int]());
  check (size_of[int]() == size_of[rec(int x)]());
}
