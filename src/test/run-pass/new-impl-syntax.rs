struct Thingy {
    x: int,
    y: int
}

impl ToStr for Thingy {
    pure fn to_str(&self) -> ~str {
        fmt!("{ x: %d, y: %d }", self.x, self.y)
    }
}

struct PolymorphicThingy<T> {
    x: T
}

impl<T:ToStr> ToStr for PolymorphicThingy<T> {
    pure fn to_str(&self) -> ~str {
        self.x.to_str()
    }
}

pub fn main() {
    io::println(Thingy { x: 1, y: 2 }.to_str());
    io::println(PolymorphicThingy { x: Thingy { x: 1, y: 2 } }.to_str());
}

