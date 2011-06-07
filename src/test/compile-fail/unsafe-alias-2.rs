// error-pattern:invalidate alias x

fn whoknows(@mutable int x) {
    *x = 10;
}

fn main() {
    auto box = @mutable 1;
    alt (*box) {
        case (?x) {
            whoknows(box);
            log_err x;
        }
    }
}
