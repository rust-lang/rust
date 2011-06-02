// -*- rust -*-

use std;
import std::io;
import std::str;

fn test_simple(str tmpfilebase) {
  let str tmpfile = tmpfilebase + ".tmp";
  log tmpfile;
  let str frood = "A hoopy frood who really knows where his towel is.";
  log frood;

  {
    let io::writer out = io::file_writer(tmpfile, [io::create]);
    out.write_str(frood);
  }

  let io::reader inp = io::file_reader(tmpfile);
  let str frood2 = inp.read_c_str();
  log frood2;
  assert (str::eq(frood, frood2));
}

fn main(vec[str] argv) {
  test_simple(argv.(0));
}
