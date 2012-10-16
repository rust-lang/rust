use sys::size_of;
extern mod std;

struct Cat {
    x: int
}

struct Kitty {
    x: int,
    drop {}
}

fn main() {
    assert (size_of::<Cat>() == 8 as uint);
    assert (size_of::<Kitty>() == 16 as uint);
}
