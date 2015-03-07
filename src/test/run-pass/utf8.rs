// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679
// no-pretty-expanded FIXME #15189

pub fn main() {
    let yen: char = '¥'; // 0xa5
    let c_cedilla: char = 'ç'; // 0xe7
    let thorn: char = 'þ'; // 0xfe
    let y_diaeresis: char = 'ÿ'; // 0xff
    let pi: char = 'Π'; // 0x3a0

    assert_eq!(yen as int, 0xa5);
    assert_eq!(c_cedilla as int, 0xe7);
    assert_eq!(thorn as int, 0xfe);
    assert_eq!(y_diaeresis as int, 0xff);
    assert_eq!(pi as int, 0x3a0);

    assert_eq!(pi as int, '\u{3a0}' as int);
    assert_eq!('\x0a' as int, '\n' as int);

    let bhutan: String = "འབྲུག་ཡུལ།".to_string();
    let japan: String = "日本".to_string();
    let uzbekistan: String = "Ўзбекистон".to_string();
    let austria: String = "Österreich".to_string();

    let bhutan_e: String =
        "\u{f60}\u{f56}\u{fb2}\u{f74}\u{f42}\u{f0b}\u{f61}\u{f74}\u{f63}\u{f0d}".to_string();
    let japan_e: String = "\u{65e5}\u{672c}".to_string();
    let uzbekistan_e: String =
        "\u{40e}\u{437}\u{431}\u{435}\u{43a}\u{438}\u{441}\u{442}\u{43e}\u{43d}".to_string();
    let austria_e: String = "\u{d6}sterreich".to_string();

    let oo: char = 'Ö';
    assert_eq!(oo as int, 0xd6);

    fn check_str_eq(a: String, b: String) {
        let mut i: int = 0;
        for ab in a.bytes() {
            println!("{}", i);
            println!("{}", ab);
            let bb: u8 = b.as_bytes()[i as uint];
            println!("{}", bb);
            assert_eq!(ab, bb);
            i += 1;
        }
    }

    check_str_eq(bhutan, bhutan_e);
    check_str_eq(japan, japan_e);
    check_str_eq(uzbekistan, uzbekistan_e);
    check_str_eq(austria, austria_e);
}
