type point = { x: int, y: int };
type character = { pos: ~point };

fn get_x(x: &r/character) -> &r/int {
    // interesting case because the scope of this
    // borrow of the unique pointer is in fact
    // larger than the fn itself
    return &x.pos.x;
}

fn main() {
}

