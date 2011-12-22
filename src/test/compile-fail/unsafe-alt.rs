// error-pattern:invalidate reference i

tag foo { left({mutable x: int}); right(bool); }

fn main() {
    let x = left({mutable x: 10});
    alt x { left(i) { x = right(false); log_full(core::debug, i); } _ { } }
}
