//! This primarily benchmarks `impl Debug for str`,
//! and it also explicitly tests that we minimizes calls to the underlying `Write`r.
//! While that is an implementation detail and there are no guarantees about it,
//! we should still try to minimize those calls over time rather than regress them.

use std::fmt::{self, Write};

use test::{Bencher, black_box};

#[derive(Default)]
struct CountingWriter {
    buf: String,
    write_calls: usize,
}

impl Write for CountingWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.buf.push_str(s);
        self.write_calls += 1;
        Ok(())
    }
}

fn assert_fmt(s: &str, expected: &str, expected_write_calls: usize) {
    let mut w = CountingWriter::default();

    write!(&mut w, "{s:?}").unwrap();
    assert_eq!(s.len(), 64);
    assert_eq!(w.buf, expected);
    assert_eq!(w.write_calls, expected_write_calls);
}

#[bench]
fn ascii_only(b: &mut Bencher) {
    let s = "just a bit of ascii text that has no escapes. 64 bytes exactly!!";
    assert_fmt(s, r#""just a bit of ascii text that has no escapes. 64 bytes exactly!!""#, 3);
    b.iter(|| {
        black_box(format!("{:?}", black_box(s)));
    });
}

#[bench]
fn ascii_escapes(b: &mut Bencher) {
    let s = "some\tmore\tascii\ttext\nthis time with some \"escapes\", also 64 byte";
    assert_fmt(
        s,
        r#""some\tmore\tascii\ttext\nthis time with some \"escapes\", also 64 byte""#,
        15,
    );
    b.iter(|| {
        black_box(format!("{:?}", black_box(s)));
    });
}

#[bench]
fn some_unicode(b: &mut Bencher) {
    let s = "egy kis szöveg néhány unicode betűvel. legyen ez is 64 byte.";
    assert_fmt(s, r#""egy kis szöveg néhány unicode betűvel. legyen ez is 64 byte.""#, 3);
    b.iter(|| {
        black_box(format!("{:?}", black_box(s)));
    });
}

#[bench]
fn mostly_unicode(b: &mut Bencher) {
    let s = "предложение из кириллических букв.";
    assert_fmt(s, r#""предложение из кириллических букв.""#, 3);
    b.iter(|| {
        black_box(format!("{:?}", black_box(s)));
    });
}

#[bench]
fn mixed(b: &mut Bencher) {
    let s = "\"❤️\"\n\"hűha ez betű\"\n\"кириллических букв\".";
    assert_fmt(s, r#""\"❤\u{fe0f}\"\n\"hűha ez betű\"\n\"кириллических букв\".""#, 21);
    b.iter(|| {
        black_box(format!("{:?}", black_box(s)));
    });
}
