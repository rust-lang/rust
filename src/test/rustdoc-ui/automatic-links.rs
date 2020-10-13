#![deny(automatic_links)]

/// [http://a.com](http://a.com)
//~^ ERROR unneeded long form for URL
/// [http://b.com]
//~^ ERROR unneeded long form for URL
///
/// [http://b.com]: http://b.com
///
/// [http://c.com][http://c.com]
pub fn a() {}

/// https://somewhere.com?hello=12
//~^ ERROR won't be a link as is
pub fn c() {}

/// <https://somewhere.com>
/// [a](http://a.com)
/// [b]
///
/// [b]: http://b.com
pub fn everything_is_fine_here() {}
