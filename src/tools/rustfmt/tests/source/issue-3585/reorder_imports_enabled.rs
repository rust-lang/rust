// rustfmt-inline_attribute_width: 100
// rustfmt-reorder_imports: true

#[cfg(unix)]
extern crate crateb;
#[cfg(unix)]
extern crate cratea;

#[cfg(unix)]
use crateb;
#[cfg(unix)]
use cratea;
