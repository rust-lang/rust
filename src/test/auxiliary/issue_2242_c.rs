#[link(name = "c", vers = "0.1")];
#[crate_type = "lib"];

extern mod a;

use a::to_strz;

impl bool: to_strz {
    fn to_strz() -> ~str { fmt!("%b", self) }
}
