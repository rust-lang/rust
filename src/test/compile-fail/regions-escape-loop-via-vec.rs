// The type of `y` ends up getting inferred to the type of the block.
fn broken() -> int {
    let mut x = 3;
    let mut y = ~[&mut x];
    while x < 10 {
        let mut z = x;
        y += ~[&mut z]; //~ ERROR illegal borrow
        x += 1;
    }
    vec::foldl(0, y, |v, p| v + **p )
}

fn main() { }