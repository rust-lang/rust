// xfail-pretty Issue #1510

tag color {
    red = 0xff0000;
    green = 0x00ff00;
    blue = 0x0000ff;
    black = 0x000000;
    white = 0xFFFFFF;
}

fn main() {
    assert uint::to_str(red as uint, 10u) == #fmt["%?", red];
    assert uint::to_str(green as uint, 10u) == #fmt["%?", green];
    assert uint::to_str(white as uint, 10u) == #fmt["%?", white];
}

