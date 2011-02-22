// -*- rust -*-

use std;
import std.io;
import std._str;

fn test_simple(str tmpfilebase) {
  let str tmpfile = tmpfilebase + ".tmp";
  log tmpfile;
  let str frood = "A hoopy frood who really knows where his towel is.";
  log frood;

  {
    let io.buf_writer out = io.new_buf_writer(tmpfile, vec(io.create()));
    out.write(_str.bytes(frood));
  }

  let io.buf_reader inp = io.new_buf_reader(tmpfile);
  let str frood2 = _str.from_bytes(inp.read());
  log frood2;
  check (_str.eq(frood, frood2));
}

fn main(vec[str] argv) {
  test_simple(argv.(0));
}
