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
