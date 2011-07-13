// -*- rust -*-
use std;
import std::io;
import std::str;

#[cfg(target_os = "linux")]
#[cfg(target_os = "win32")]
#[test]
fn test_simple() {
    let str tmpfile = "test/run-pass/lib-io-test-simple.tmp";
    log tmpfile;
    let str frood = "A hoopy frood who really knows where his towel is.";
    log frood;
    {
        let io::writer out = io::file_writer(tmpfile, [io::create,
                                                       io::truncate]);
        out.write_str(frood);
    }
    let io::reader inp = io::file_reader(tmpfile);
    let str frood2 = inp.read_c_str();
    log frood2;
    assert (str::eq(frood, frood2));
}

// FIXME (726)
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn test_simple() {}

