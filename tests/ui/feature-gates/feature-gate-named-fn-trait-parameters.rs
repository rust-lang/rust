fn parse_my_data(
    data: &str,
    log: impl Fn(msg: String),
    //~^ ERROR `Trait(...)` syntax does not support named parameters
) { }

fn main() {}
