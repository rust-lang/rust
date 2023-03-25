// run-pass

pub fn main() {
    let a = 2;
    let b = 4;
    let _c = f"a ({a}) + b ({b}) = {a + b}";
    assert_eq!(_c, "a (2) + b (4) = 6");
}
