// error-pattern:invalidate reference x

fn whoknows(x: @mutable {mutable x: int}) { x.x = 10; }

fn main() {
    let box = @mutable {mutable x: 1};
    alt *box { x { whoknows(box); log_full(core::error, x); } }
}
