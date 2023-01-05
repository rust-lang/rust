// run-pass
trait Stringify {
    fn to_string(&self) -> String;
}

impl Stringify for u32 {
    fn to_string(&self) -> String { format!("u32: {}", *self) }
}

impl Stringify for f32 {
    fn to_string(&self) -> String { format!("f32: {}", *self) }
}

fn print<T: Stringify>(x: T) -> String {
    x.to_string()
}

fn main() {
    assert_eq!(&print(5), "u32: 5");
    assert_eq!(&print(5.0), "f32: 5");
}
