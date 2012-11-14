struct Font {
    fontbuf: uint,
    cairo_font: uint,
    font_dtor: uint,

}

impl Font : Drop {
    fn finalize() {}
}

fn Font() -> Font {
    Font {
        fontbuf: 0,
        cairo_font: 0,
        font_dtor: 0
    }
}

fn main() {
    let _f = @Font();
}
