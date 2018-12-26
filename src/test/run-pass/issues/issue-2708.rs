// run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

// pretty-expanded FIXME #23616

#![feature(box_syntax)]

struct Font {
    fontbuf: usize,
    cairo_font: usize,
    font_dtor: usize,

}

impl Drop for Font {
    fn drop(&mut self) {}
}

fn Font() -> Font {
    Font {
        fontbuf: 0,
        cairo_font: 0,
        font_dtor: 0
    }
}

pub fn main() {
    let _f: Box<_> = box Font();
}
