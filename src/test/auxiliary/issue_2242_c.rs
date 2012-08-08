#[link(name = "c", vers = "0.1")];
#[crate_type = "lib"];

use a;

import a::to_str;

impl bool: to_str {
    fn to_str() -> ~str { fmt!{"%b", self} }
}
