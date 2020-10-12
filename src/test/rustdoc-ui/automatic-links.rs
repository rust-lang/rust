#![deny(automatic_links)]

/// [http://a.com](http://a.com)
//~^ ERROR Unneeded long form for URL
/// [http://b.com]
//~^ ERROR Unneeded long form for URL
///
/// [http://b.com]: http://b.com
///
/// [http://c.com][http://c.com]
pub fn a() {}

/// [a](http://a.com)
/// [b]
///
/// [b]: http://b.com
pub fn everything_is_fine_here() {}
