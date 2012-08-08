#[link(name = "a", vers = "0.1")];
#[crate_type = "lib"];

trait to_str {
    fn to_str() -> ~str;
}

impl ~str: to_str {
    fn to_str() -> ~str { self }
}
