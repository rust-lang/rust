// error-pattern:invalidate alias x

fn whoknows(x: @mutable int) { *x = 10; }

fn main() {
    let box = @mutable 1;
    alt *box { x { whoknows(box); log_err x; } }
}
