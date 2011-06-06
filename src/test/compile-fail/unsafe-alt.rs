// error-pattern:x is being aliased

tag foo {
    left(int);
    right(bool);
}

fn main() {
    auto x = left(10);
    alt (x) {
        case (left(?i)) {
            x = right(false);
        }
        case (_) {}
    }
}
