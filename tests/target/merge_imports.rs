// rustfmt-merge_imports: true

use a::{a, b, c, d, e, f, g};

#[doc(hidden)]
use a::b;
use a::{c, d};

#[doc(hidden)]
use a::b;
use a::{c, d, e};

use foo::{a, b, c};
pub use foo::{bar, foobar};

use a::b::c::{d, xxx, yyy, zzz, *};
