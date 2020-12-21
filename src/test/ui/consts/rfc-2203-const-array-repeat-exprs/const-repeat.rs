// run-pass

// Repeating a *constant* of non-Copy type (not just a constant expression) is already stable.

const EMPTY: Vec<i32> = Vec::new();

pub fn bar() -> [Vec<i32>; 2] {
    [EMPTY; 2]
}

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        panic!("BOOM!");
    }
}

const BOOM: Bomb = Bomb;

fn main() {
    let _x = bar();

    // Make sure the destructor does not get called for empty arrays. `[CONST; N]` should
    // instantiate (and then later drop) the const exactly `N` times.
    let _x = [BOOM; 0];
}
