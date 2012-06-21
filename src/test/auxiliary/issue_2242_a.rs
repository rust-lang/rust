#[link(name = "a", vers = "0.1")];
#[crate_type = "lib"];

iface to_str {
    fn to_str() -> str;
}

impl of to_str for str {
    fn to_str() -> str { self }
}
