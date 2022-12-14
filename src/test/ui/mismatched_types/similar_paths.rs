enum Option<T> {
    Some(T),
    None,
}

pub fn foo() -> Option<u8> {
    Some(42_u8)
    //~^ ERROR mismatched types [E0308]
}

fn main() {}
