// error-pattern:invalidate reference i

tag foo { left(int); right(bool); }

fn main() {
    let x = left(10);
    alt x { left(i) { x = right(false); log i; } _ { } }
}
