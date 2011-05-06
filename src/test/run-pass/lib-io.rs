// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

use std;
import std.IO;
import std.Str;

fn test_simple(str tmpfilebase) {
  let str tmpfile = tmpfilebase + ".tmp";
  log tmpfile;
  let str frood = "A hoopy frood who really knows where his towel is.";
  log frood;

  {
    let IO.writer out = IO.file_writer(tmpfile, vec(IO.create));
    out.write_str(frood);
  }

  let IO.reader inp = IO.file_reader(tmpfile);
  let str frood2 = inp.read_c_str();
  log frood2;
  assert (Str.eq(frood, frood2));
}

fn main(vec[str] argv) {
  test_simple(argv.(0));
}
