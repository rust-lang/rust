// error-pattern:invalidate alias i

tag foo {
    left(int);
    right(bool);
}

fn main() {
    auto x = left(10);
    alt (x) {
        case (left(?i)) {
            x = right(false);
            log i;
        }
        case (_) {}
    }
}
