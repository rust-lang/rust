fn parse_my_data(
    data: &str,
    log: impl Fn(msg: String),
    //~^ ERROR named parameters in parenthesized generic argument lists are experimental [E0658]
) { }

fn main() {}
