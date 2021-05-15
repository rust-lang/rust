// rustfmt-imports_granularity: Crate

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

// https://github.com/rust-lang/rustfmt/issues/3808
use d::{self};
use e::{self as foo};
use f::{self, b};
use g::a;
use g::{self, b};
use h::{a};
use i::a::{self};
use j::{a::{self}};

use {k::{a, b}, l::{a, b}};
use {k::{c, d}, l::{c, d}};
