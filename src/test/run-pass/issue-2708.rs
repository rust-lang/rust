class Font {
    let fontbuf: uint;
    let cairo_font: uint;
    let font_dtor: uint;

    new() {
        self.fontbuf = 0;
        self.cairo_font = 0;
        self.font_dtor = 0;
    }

    drop { }
}

fn main() {
    let _f = @Font();
}
