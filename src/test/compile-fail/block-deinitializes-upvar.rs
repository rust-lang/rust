// error-pattern:moving out of immutable upvar
fn force(f: fn()) { f(); }
fn main() {
    let mut x = @{x: 17, y: 2};
    let y = @{x: 5, y: 5};

    force({|| x <- y;});
}
