// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate test;
use std::fmt;
use std::io::MemWriter;
use super::{escape, unescape};
use super::escape::{EscapeWriter, UnescapeWriter};
use super::escape::{EscapeDefault, EscapeText, EscapeAttr, EscapeSingleQuoteAttr};

struct Test(StrBuf);

impl fmt::Show for Test {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Test(ref s) = *self;
        write!(fmt, "<Test>{}</Test>", s)
    }
}

struct UnTest(&'static str, &'static str);

impl fmt::Show for UnTest {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let UnTest(s1, s2) = *self;
        try!(write!(fmt, "{}", s1));
        write!(fmt, "{}", s2)
    }
}

#[test]
fn test_escape() {
    let s = r#"<script src="evil.domain?foo&" type='baz'>"#;
    assert_eq!(escape(s).as_slice(), "&lt;script src=&quot;evil.domain?foo&amp;&quot; \
                                        type=&#39;baz&#39;&gt;");

    let t = Test("foo".to_strbuf());
    assert_eq!(escape(t), "&lt;Test&gt;foo&lt;/Test&gt;".to_owned());
}

#[test]
fn test_unescape() {
    let s = "&lt;script src=&quot;evil.domain?foo&amp;&quot; type=&#39;baz&#39;&gt;";
    assert_eq!(unescape(s), r#"<script src="evil.domain?foo&" type='baz'>"#.to_owned());

    assert_eq!(unescape("&rarr;"), "\u2192".to_owned());
    assert_eq!(unescape("&&amp;amp;amp;"), "&&amp;amp;".to_owned());
    assert_eq!(unescape("&CounterClockwiseContourIntegral;"), "\u2233".to_owned());
    assert_eq!(unescape("&amp"), "&".to_owned());
    assert_eq!(unescape(UnTest("&am", "p;")), "&".to_owned());
    assert_eq!(unescape("&fakentity"), "&fakentity".to_owned());
    assert_eq!(unescape("&fakentity;"), "&fakentity;".to_owned());
    assert_eq!(unescape("&aeligtest"), "ætest".to_owned());
    assert_eq!(unescape("&#0abc"), "\uFFFDabc".to_owned());
    assert_eq!(unescape("&#abc"), "&#abc".to_owned());
    assert_eq!(unescape("&#xgabc"), "&#xgabc".to_owned());
    assert_eq!(unescape("&#X2022; &#XYZ;"), "\u2022 &#XYZ;".to_owned());
    // this next escape overflows a u64. WebKit incorrectly treats this as &#x2022;
    assert_eq!(unescape("&#x100000000000000002022;"), "\uFFFD".to_owned());
    assert_eq!(unescape("&#x80;&#x81;&#x9F;&#0;"), "\u20AC\x81\u0178\uFFFD".to_owned());
}

macro_rules! escape_test{
    ($mode:ident, $input:expr, $result:expr) => {{
        use std::path::BytesContainer;
        let mode = concat_idents!(Escape, $mode);
        let mut w = EscapeWriter::new(MemWriter::new(), mode);
        w.write($input.container_as_bytes()).unwrap();
        let v = w.unwrap().unwrap();
        // provide better errors by comparing strings when possible
        let result = $result;
        match (StrBuf::from_utf8(v), result.container_as_str()) {
            (Ok(s), Some(res)) => assert_eq!(s.as_slice(), res),
            (Ok(s), None) => assert_eq!(s.as_bytes(), result.container_as_bytes()),
            (Err(v), _) => assert_eq!(v.as_slice(), result.container_as_bytes())
        }
    }}
}

#[test]
fn test_escapewriter_default() {
    escape_test!(Default, "<>&\"'abc()\u2022", "&lt;&gt;&amp;&quot;&#39;abc()\u2022");
    escape_test!(Default, "", "");
    escape_test!(Default, bytes!(0, 1, 0x80, "\x80"), bytes!(0, 1, 0x80, "\x80"));
}

#[test]
fn test_escapewriter_text() {
    escape_test!(Text, "<>&\"'abc()\u2022", "&lt;&gt;&amp;\"'abc()\u2022");
    escape_test!(Text, "", "");
    escape_test!(Text, bytes!(0, 1, 0x80, "\x80"), bytes!(0, 1, 0x80, "\x80"));
}

#[test]
fn test_escapewriter_attr() {
    escape_test!(Attr, "<>&\"'abc()\u2022", "<>&amp;&quot;'abc()\u2022");
    escape_test!(Attr, "", "");
    escape_test!(Attr, bytes!(0, 1, 0x80, "\x80"), bytes!(0, 1, 0x80, "\x80"));
}

#[test]
fn test_escapewriter_singlequote_attr() {
    escape_test!(SingleQuoteAttr, "<>&\"'abc()\u2022", "<>&amp;\"&#39;abc()\u2022");
    escape_test!(SingleQuoteAttr, "", "");
    escape_test!(SingleQuoteAttr, bytes!(0, 1, 0x80, "\x80"), bytes!(0, 1, 0x80, "\x80"));
}

#[test]
fn test_roundtrip_writer() {
    let mut w = EscapeWriter::new(MemWriter::new(), EscapeDefault);
    w.write_str("<>&\"'abc()\u2022").unwrap();
    w.write(bytes!(0, 1, 0x80, "\x80")).unwrap();
    let v = w.unwrap().unwrap();
    let mut w = UnescapeWriter::new(MemWriter::new());
    w.write(v.as_slice()).unwrap();
    let v = w.unwrap().unwrap();
    assert_eq!(v.as_slice(), bytes!("<>&\"'abc()\u2022", 0, 1, 0x80, "\x80"));
}

#[test]
fn test_unescapewriter_with_allowed_char() {
    let mut w = UnescapeWriter::with_allowed_char(MemWriter::new(), 'q');
    w.write_str("&lt;&gt;&quot;").unwrap();
    let v = w.unwrap().unwrap();
    assert_eq!(v.as_slice(), "<>&quot;".as_bytes());
}

// Tests from python's html module
// See http://hg.python.org/cpython/file/82caec3865e3/Lib/test/test_html.py
mod python {
    use {escape, unescape};
    use escape::{EscapeWriter, EscapeText};
    use std::io::MemWriter;

    #[test]
    fn test_escape() {
        // python converts ' to &#x27; but we go to &#39;
        assert_eq!(escape(r#"'<script>"&foo;"</script>'"#).as_slice(),
                   "&#39;&lt;script&gt;&quot;&amp;foo;&quot;&lt;/script&gt;&#39;");
        let mut w = EscapeWriter::new(MemWriter::new(), EscapeText);
        assert!(w.write_str(r#"'<script>"&foo;"</script>'"#).is_ok());
        assert_eq!(w.unwrap().unwrap().as_slice(),
                   r#"'&lt;script&gt;"&amp;foo;"&lt;/script&gt;'"#.as_bytes());
    }

    #[test]
    fn test_unescape() {
        macro_rules! check{
            ($text:expr, $exp:expr) => {
                assert_eq!(unescape($text).as_slice(), $exp.as_slice());
            };
            (num: $num:expr, $exp:expr) => {{
                let num = $num;
                let exp = $exp;
                check!(format!(r"&\#{}", num), exp);
                check!(format!(r"&\#{};", num), exp);
                check!(format!(r"&\#x{:x}", num), exp);
                check!(format!(r"&\#x{:x};", num), exp);
            }};
        }

        // check text with no character references
        check!("no character references", "no character references");
        // check & followed by invalid chars
        check!("&\n&\t& &&", "&\n&\t& &&");
        // check & followed by numbers and letters
        check!("&0 &9 &a &0; &9; &a;", "&0 &9 &a &0; &9; &a;");
        // check incomplete entities at the end of the string
        for x in ["&", "&#", "&#x", "&#X", "&#y", "&#xy", "&#Xy"].iter() {
            check!(x, x);
            check!(x+";", x+";");
        }
        // check several combinations of numeric character references,
        // possibly followed by different characters
        // NB: no runtime formatting strings so the loop has been unrolled
        for (&num, &c) in [65u32, 97, 34, 38, 0x2603, 0x101234].iter()
                          .zip(["A", "a", "\"", "&", "\u2603", "\U00101234"].iter()) {
            let v = [format!(r"&\#{}",num),     format!(r"&\#{:07}",num),
                     format!(r"&\#{};",num),    format!(r"&\#{:07};",num),
                     format!(r"&\#x{:x}",num),  format!(r"&\#x{:06x}",num),
                     format!(r"&\#x{:x};",num), format!(r"&\#x{:06x};",num),
                     format!(r"&\#x{:X}",num),  format!(r"&\#x{:06X}",num),
                     format!(r"&\#X{:x};",num), format!(r"&\#X{:06x};",num)];
            for s in v.iter() {
                check!(s.as_slice(), c);
                for end in [" ", "X"].iter() {
                    check!(*s+*end, c+*end);
                }
            }
        }
        // check invalid codepoints
        for &cp in [0xD800, 0xDB00, 0xDC00, 0xDFFF, 0x110000].iter() {
            check!(num: cp, "\uFFFD");
        }
        // check more invalid codepoints
        // this test is elided because it's wrong. I don't know why cpython thinks codepoints
        // [0x1, 0xb, 0xe, 0x7f, 0xfffe, 0xffff, 0x10fffe, 0x10ffff] should return nothing.
        // check invalid numbers
        for (&num, &c) in [0x0d, 0x80, 0x95, 0x9d].iter()
                          .zip(["\r", "\u20ac", "\u2022", "\x9d"].iter()) {
            check!(num: num, c);
        }
        // check small numbers
        check!(num: 0, "\uFFFD");
        check!(num: 9, "\t");
        // check a big number
        check!(num: 1000000000000000000u64, "\uFFFD");
        // check that multiple trailing semicolons are handled correctly
        for e in ["&quot;;", "&#34;;", "&#x22;;", "&#X22;;"].iter() {
            check!(*e, "\";");
        }
        // check that semicolons in the middle don't create problems
        for e in ["&quot;quot;", "&#34;quot;", "&#x22;quot;", "&#X22;quot;"].iter() {
            check!(*e, "\"quot;");
        }
        // check triple adjacent charrefs
        for e in ["&quot", "&#34", "&#x22", "&#X22"].iter() {
            check!(e.repeat(3), r#"""""#);
            check!((*e+";").repeat(3), r#"""""#);
        }
        // check that the case is respected
        for e in ["&amp", "&amp;", "&AMP", "&AMP;"].iter() {
            check!(*e, "&");
        }
        for e in ["&Amp", "&Amp;"].iter() {
            check!(*e, *e);
        }
        // check that non-existent named entities are returned unchanged
        check!("&svadilfari;", "&svadilfari;");
        // the following examples are in the html5 specs
        check!("&notit", "¬it");
        check!("&notit;", "¬it;");
        check!("&notin", "¬in");
        check!("&notin;", "∉");
        // a similar example with a long name
        check!("&notReallyAnExistingNamedCharacterReference;",
               "¬ReallyAnExistingNamedCharacterReference;");
        // longest valid name
        check!("&CounterClockwiseContourIntegral;", "∳");
        // check a charref that maps to two unicode chars
        check!("&acE;", "\u223E\u0333");
        check!("&acE", "&acE");
        // test a large number of entities
        check!("&#123; ".repeat(1050), "{ ".repeat(1050));
        // check some html5 entities
        check!("&Eacuteric&Eacute;ric&alphacentauri&alpha;centauri",
               "ÉricÉric&alphacentauriαcentauri");
        check!("&co;", "&co;");
    }
}

#[bench]
fn bench_escape(b: &mut test::Bencher) {
    let s = "<script src=\"evil.domain?foo&\" type='baz'>";
    b.iter(|| escape(s));
}

#[bench]
fn bench_unescape(b: &mut test::Bencher) {
    let s = "&lt;script src=&quot;evil.domain?foo&amp;&quot; type=&#39;baz&#39;&gt;";
    b.iter(|| unescape(s));
}

#[bench]
fn bench_longest_entity(b: &mut test::Bencher) {
    let s = "&CounterClockwiseContourIntegral;";
    b.iter(|| assert_eq!(unescape(s).as_slice(), "\u2233"));
}

#[bench]
fn bench_longest_non_entity(b: &mut test::Bencher) {
    let s = "&CounterClockwiseContourIntegraX;";
    b.iter(|| assert_eq!(unescape(s).as_slice(), "&CounterClockwiseContourIntegraX;"));
}

#[bench]
fn bench_short_entity_long_tail(b: &mut test::Bencher) {
    let s = "&ampnterClockwiseContourIntegral";
    b.iter(|| assert_eq!(unescape(s).as_slice(), "&nterClockwiseContourIntegral"));
}
