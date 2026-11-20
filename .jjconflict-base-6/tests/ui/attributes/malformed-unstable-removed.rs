#![feature(staged_api)]

#![unstable_removed(feature = "old_feature")]
//~^ ERROR: malformed `unstable_removed` attribute

#![unstable_removed(invalid = "old_feature")]
//~^ ERROR: malformed `unstable_removed` attribute

#![unstable_removed("invalid literal")]
//~^ ERROR: malformed `unstable_removed` attribute

#![unstable_removed = "invalid literal"]
//~^ ERROR: malformed `unstable_removed` attribute

#![stable(feature="main", since="1.0.0")]
fn main() {}
