import core::*;

// -*- rust -*-
use std;
import std::io;
import io::{writer_util, reader_util};
import str;
import result;

#[test]
fn test_simple() {
    let tmpfile: str = "tmp/lib-io-test-simple.tmp";
    log(debug, tmpfile);
    let frood: str = "A hoopy frood who really knows where his towel is.";
    log(debug, frood);
    {
        let out: io::writer =
            result::get(io::file_writer(tmpfile, [io::create, io::truncate]));
        out.write_str(frood);
    }
    let inp: io::reader = result::get(io::file_reader(tmpfile));
    let frood2: str = inp.read_c_str();
    log(debug, frood2);
    assert (str::eq(frood, frood2));
}

#[test]
fn test_readchars_empty() {
    let inp : io::reader = io::string_reader("");
    let res : [char] = inp.read_chars(128u);
    assert(vec::len(res) == 0u);
}

#[test]
fn test_readchars_wide() {
    let wide_test = "生锈的汤匙切肉汤hello生锈的汤匙切肉汤";
    let ivals : [int] = [
        29983, 38152, 30340, 27748,
        21273, 20999, 32905, 27748,
        104, 101, 108, 108, 111,
        29983, 38152, 30340, 27748,
        21273, 20999, 32905, 27748];
    fn check_read_ln(len : uint, s: str, ivals: [int]) {
        let inp : io::reader = io::string_reader(s);
        let res : [char] = inp.read_chars(len);
        if (len <= vec::len(ivals)) {
            assert(vec::len(res) == len);
        }
        assert(vec::slice(ivals, 0u, vec::len(res)) ==
               vec::map(res, {|x| x as int}));
    }
    let i = 0u;
    while i < 8u {
        check_read_ln(i, wide_test, ivals);
        i += 1u;
    }
    // check a long read for good measure
    check_read_ln(128u, wide_test, ivals);
}

#[test]
fn test_readchar() {
    let inp : io::reader = io::string_reader("生");
    let res : char = inp.read_char();
    assert(res as int == 29983);
}

#[test]
fn test_readchar_empty() {
    let inp : io::reader = io::string_reader("");
    let res : char = inp.read_char();
    assert(res as int == -1);
}

#[test]
fn file_reader_not_exist() {
    alt io::file_reader("not a file") {
      result::err(e) {
        assert e == "error opening not a file";
      }
      result::ok(_) { fail; }
    }
}

#[test]
fn file_writer_bad_name() {
    alt io::file_writer("?/?", []) {
      result::err(e) {
        assert e == "error opening ?/?";
      }
      result::ok(_) { fail; }
    }
}

#[test]
fn buffered_file_writer_bad_name() {
    alt io::buffered_file_writer("?/?") {
      result::err(e) {
        assert e == "error opening ?/?";
      }
      result::ok(_) { fail; }
    }
}
