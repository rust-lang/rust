import back.x86;
import std._str;
import std._vec;
import std.os.libc;

fn main(vec[str] args) {
  auto module_asm = x86.get_module_asm() + "\n";
  auto bytes = _str.bytes(module_asm);
  auto b = _vec.buf[u8](bytes);
  libc.write(1, b, _vec.len[u8](bytes));
}
