struct Thing {
    x: u32,
    y: u32
}

fn main() {
    let thing = Thing { x: 0, y: 0 };
    match thing {
        Thing { x, y, z } => {}
        //~^ ERROR struct `Thing` does not have a field named `z` [E0026]
    }
}
