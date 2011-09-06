

// -*- rust -*-

use std;
import std::sha1;
import std::vec;
import std::str;

#[test]
fn test() {
    type test = {input: str, output: [u8]};

    fn a_million_letter_a() -> str {
        let i = 0;
        let rs = "";
        while i < 100000 { rs += "aaaaaaaaaa"; i += 1; }
        ret rs;
    }
    // Test messages from FIPS 180-1

    let fips_180_1_tests: [test] =
        [{input: "abc",
          output:
              [0xA9u8, 0x99u8, 0x3Eu8, 0x36u8, 0x47u8, 0x06u8, 0x81u8, 0x6Au8,
               0xBAu8, 0x3Eu8, 0x25u8, 0x71u8, 0x78u8, 0x50u8, 0xC2u8, 0x6Cu8,
               0x9Cu8, 0xD0u8, 0xD8u8, 0x9Du8]},
         {input:
              "abcdbcdecdefdefgefghfghighij" + "hijkijkljklmklmnlmnomnopnopq",
          output:
              [0x84u8, 0x98u8, 0x3Eu8, 0x44u8, 0x1Cu8, 0x3Bu8, 0xD2u8, 0x6Eu8,
               0xBAu8, 0xAEu8, 0x4Au8, 0xA1u8, 0xF9u8, 0x51u8, 0x29u8, 0xE5u8,
               0xE5u8, 0x46u8, 0x70u8, 0xF1u8]},
         {input: a_million_letter_a(),
          output:
              [0x34u8, 0xAAu8, 0x97u8, 0x3Cu8, 0xD4u8, 0xC4u8, 0xDAu8, 0xA4u8,
               0xF6u8, 0x1Eu8, 0xEBu8, 0x2Bu8, 0xDBu8, 0xADu8, 0x27u8, 0x31u8,
               0x65u8, 0x34u8, 0x01u8, 0x6Fu8]}];
    // Examples from wikipedia

    let wikipedia_tests: [test] =
        [{input: "The quick brown fox jumps over the lazy dog",
          output:
              [0x2fu8, 0xd4u8, 0xe1u8, 0xc6u8, 0x7au8, 0x2du8, 0x28u8, 0xfcu8,
               0xedu8, 0x84u8, 0x9eu8, 0xe1u8, 0xbbu8, 0x76u8, 0xe7u8, 0x39u8,
               0x1bu8, 0x93u8, 0xebu8, 0x12u8]},
         {input: "The quick brown fox jumps over the lazy cog",
          output:
              [0xdeu8, 0x9fu8, 0x2cu8, 0x7fu8, 0xd2u8, 0x5eu8, 0x1bu8, 0x3au8,
               0xfau8, 0xd3u8, 0xe8u8, 0x5au8, 0x0bu8, 0xd1u8, 0x7du8, 0x9bu8,
               0x10u8, 0x0du8, 0xb4u8, 0xb3u8]}];
    let tests = fips_180_1_tests + wikipedia_tests;
    fn check_vec_eq(v0: &[u8], v1: &[u8]) {
        assert (vec::len::<u8>(v0) == vec::len::<u8>(v1));
        let len = vec::len::<u8>(v0);
        let i = 0u;
        while i < len {
            let a = v0[i];
            let b = v1[i];
            assert (a == b);
            i += 1u;
        }
    }
    // Test that it works when accepting the message all at once

    let sh = sha1::mk_sha1();
    for t: test in tests {
        sh.input_str(t.input);
        let out = sh.result();
        check_vec_eq(t.output, out);
        sh.reset();
    }


    // Test that it works when accepting the message in pieces
    for t: test in tests {
        let len = str::byte_len(t.input);
        let left = len;
        while left > 0u {
            let take = (left + 1u) / 2u;
            sh.input_str(str::substr(t.input, len - left, take));
            left = left - take;
        }
        let out = sh.result();
        check_vec_eq(t.output, out);
        sh.reset();
    }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
