// error-pattern:Tried to deinitialize a variable declared in a different
fn force(f: block()) { f(); }
fn main() {
    let x = @{x: 17, y: 2};
    let y = @{x: 5, y: 5};

    force({|| x <- y;});
}
