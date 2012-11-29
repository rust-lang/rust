#[legacy_exports];

use syntax::diagnostic;

export diagnostic;

export driver;
export session;

#[legacy_exports]
mod driver;
#[legacy_exports]
mod session;
