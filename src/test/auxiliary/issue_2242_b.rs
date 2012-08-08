#[link(name = "b", vers = "0.1")];
#[crate_type = "lib"];

use a;
import a::to_str;

impl int: to_str {
    fn to_str() -> ~str { fmt!{"%?", self} }
}
