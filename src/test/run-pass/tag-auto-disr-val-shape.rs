enum color { red, green, blue, black, white, }

fn main() {
    // Ideally we would print the name of the variant, not the number
    assert (uint::to_str(red as uint, 10u) == #fmt["%?", red]);
    assert (uint::to_str(green as uint, 10u) == #fmt["%?", green]);
    assert (uint::to_str(white as uint, 10u) == #fmt["%?", white]);
}

