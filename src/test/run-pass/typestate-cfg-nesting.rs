

fn f() {
    auto x = 10;
    auto y = 11;
    if (true) { alt (x) { case (_) { y = x; } } } else { }
}

fn main() {
    auto x = 10;
    auto y = 11;
    if (true) { while (false) { y = x; } } else { }
}