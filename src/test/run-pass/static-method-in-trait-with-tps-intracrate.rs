trait Deserializer {
    fn read_int() -> int;
}

trait Deserializable<D: Deserializer> {
    static fn deserialize(d: &D) -> self;
}

impl<D: Deserializer> int: Deserializable<D> {
    static fn deserialize(d: &D) -> int {
        return d.read_int();
    }
}

struct FromThinAir { dummy: () }

impl FromThinAir: Deserializer {
    fn read_int() -> int { 22 }
}

fn main() {
    let d = FromThinAir { dummy: () };
    let i: int = deserialize(&d);
    assert i == 22;
}