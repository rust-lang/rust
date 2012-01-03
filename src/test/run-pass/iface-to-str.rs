iface to_str {
    fn to_str() -> str;
}

impl of to_str for int {
    fn to_str() -> str { int::str(self) }
}

impl <T: to_str> of to_str for [T] {
    fn to_str() -> str {
        "[" + str::connect(vec::map(self, {|e| e.to_str()}), ", ") + "]"
    }
}

fn main() {
    assert 1.to_str() == "1";
    assert [2, 3, 4].to_str() == "[2, 3, 4]";

    fn indirect<T: to_str>(x: T) -> str {
        x.to_str() + "!"
    }
    assert indirect([10, 20]) == "[10, 20]!";

    fn indirect2<T: to_str>(x: T) -> str {
        indirect(x)
    }
    assert indirect2([1]) == "[1]!";
}
