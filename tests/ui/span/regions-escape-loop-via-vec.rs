// The type of `y` ends up getting inferred to the type of the block.
fn broken() {
    let mut x = 3;
    let mut _y = vec![&mut x];
    while x < 10 {
        let mut z = x;
        _y.push(&mut z);
        //~^ ERROR `z` does not live long enough
        x += 1;
    }
}

fn main() { }
