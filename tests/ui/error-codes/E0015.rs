fn create_some() -> Option<u8> {
    Some(1)
}

const FOO: Option<u8> = create_some();
//~^ ERROR cannot call non-const function `create_some` in constants [E0015]

fn main() {}
