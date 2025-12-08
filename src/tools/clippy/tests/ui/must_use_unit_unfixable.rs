#[cfg_attr(true, must_use, deprecated)]
fn issue_12320() {}
//~^ must_use_unit

#[cfg_attr(true, deprecated, doc = "foo", must_use)]
fn issue_12320_2() {}
//~^ must_use_unit

fn main() {}
