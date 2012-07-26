type point = { x: int, y: int };
type character = { pos: ~point };

fn get_x(x: &character) -> &int {
    // interesting case because the scope of this
    // borrow of the unique pointer is in fact
    // larger than the fn itself
    ret &x.pos.x;
}

fn main() {
}

