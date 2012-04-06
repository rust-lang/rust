// The type of `y` ends up getting inferred to the type of the block.
// This generates a ton of error msgs at the moment.
fn broken() -> int {
    let mut x = 3;
    let mut y = [&x]; //! ERROR reference escapes its block
    while x < 10 {
        let z = x;
        y += [&z];
        x += 1;
    }
    vec::foldl(0, y) {|v, p| v + *p }
    //!^ ERROR reference escapes its block
    //!^^ ERROR reference escapes its block
    //!^^^ ERROR reference escapes its block
    //!^^^^ ERROR reference escapes its block
    //!^^^^^ ERROR reference escapes its block
}

fn main() { }