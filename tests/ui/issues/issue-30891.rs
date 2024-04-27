//@ run-pass
const ERROR_CONST: bool = true;

fn get() -> bool {
    false || ERROR_CONST
}

pub fn main() {
    assert_eq!(get(), true);
}
