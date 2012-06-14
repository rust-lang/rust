// xfail-fast

#[link(name = "b", vers = "0.1")];
#[crate_type = "lib"];

use a;

import a::t2;
export t2;
