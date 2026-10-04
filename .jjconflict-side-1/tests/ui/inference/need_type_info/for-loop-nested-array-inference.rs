//! Regression test for <https://github.com/rust-lang/rust/issues/51116>.
//! This used to leak internal `__next` ident into suggestion.

fn main() {
    let tiles = Default::default();
    for row in &mut tiles {
        for tile in row {
            *tile = 0;
            //~^ ERROR type annotations needed
            //~| NOTE cannot infer type
        }
    }

    let tiles: [[usize; 3]; 3] = tiles;
}
