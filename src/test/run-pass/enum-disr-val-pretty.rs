// pp-exact

enum color { red = 1, green, blue, imaginary = -1, }

fn main() {
    test_color(red, 1, "red");
    test_color(green, 2, "green");
    test_color(blue, 3, "blue");
    test_color(imaginary, -1, "imaginary");
}

fn test_color(color: color, val: int, name: str) {
    assert (color as int == val);
    assert (color as float == val as float);
}

