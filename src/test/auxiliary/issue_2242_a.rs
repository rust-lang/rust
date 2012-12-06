#[link(name = "a", vers = "0.1")];
#[crate_type = "lib"];

trait to_strz {
    fn to_strz() -> ~str;
}

impl ~str: to_strz {
    fn to_strz() -> ~str { copy self }
}
