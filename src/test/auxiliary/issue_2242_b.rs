#[link(name = "b", vers = "0.1")];
#[crate_type = "lib"];

extern mod a;
use a::to_strz;

impl int: to_strz {
    fn to_strz() -> ~str { fmt!("%?", self) }
}
