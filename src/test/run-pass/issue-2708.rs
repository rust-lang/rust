struct Font {
    let fontbuf: uint;
    let cairo_font: uint;
    let font_dtor: uint;

    drop { }
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
