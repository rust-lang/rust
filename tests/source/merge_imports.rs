// rustfmt-merge_imports: true

use a::{c,d,b};
use a::{d, e, b, a, f};
use a::{f, g, c};

#[doc(hidden)]
use a::b;
use a::c;
use a::d;

use a::{c, d, e};
#[doc(hidden)]
use a::b;
use a::d;

pub use foo::bar;
use foo::{a, b, c};
pub use foo::foobar;

use a::{b::{c::*}};
use a::{b::{c::{}}};
use a::{b::{c::d}};
use a::{b::{c::{xxx, yyy, zzz}}};
