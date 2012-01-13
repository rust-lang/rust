// xfail-test

enum color { red; green; blue; black; white; }

fn main() {
    assert (uint::to_str(red as uint, 10) == #fmt["%?", red]);
    assert (uint::to_str(green as uint, 10) == #fmt["%?", green]);
    assert (uint::to_str(white as uint, 10) == #fmt["%?", white]);
}

