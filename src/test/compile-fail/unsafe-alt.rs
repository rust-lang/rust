// error-pattern:invalidate reference i

enum foo { left({mut x: int}), right(bool) }

fn main() {
    let mut x = left({mut x: 10});
    alt x { left(i) { x = right(false); copy x; log(debug, i); } _ { } }
}
