// -*- rust -*-
use std;
import std::io;
import std::istr;

#[cfg(target_os = "linux")]
#[cfg(target_os = "win32")]
#[test]
fn test_simple() {
    let tmpfile: istr = ~"test/run-pass/lib-io-test-simple.tmp";
    log tmpfile;
    let frood: istr = ~"A hoopy frood who really knows where his towel is.";
    log frood;
    {
        let out: io::writer =
            io::file_writer(tmpfile, [io::create, io::truncate]);
        out.write_str(frood);
    }
    let inp: io::reader = io::file_reader(tmpfile);
    let frood2: istr = inp.read_c_str();
    log frood2;
    assert (istr::eq(frood, frood2));
}

// FIXME (726)
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn test_simple() { }

