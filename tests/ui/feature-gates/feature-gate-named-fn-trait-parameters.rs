fn parse_my_data(
    data: &str,
    log: impl Fn(msg: String),
    //~^ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental [E0658]
    attrs: impl Fn(#[cfg(false)] String),
    //~^ ERROR parameters in parenthesized generic argument lists with names, attributes, or c-variadics are experimental [E0658]
) { }

fn main() {}
