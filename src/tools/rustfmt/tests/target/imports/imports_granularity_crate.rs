// rustfmt-imports_granularity: Crate

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

// https://github.com/rust-lang/rustfmt/issues/3808
use d::{self};
use e::{self as foo};
use f::{self, b};
use g::{self, a, b};
use h::a;
use i::a::{self};
use j::a::{self};

use k::{a, b, c, d};
use l::{a, b, c, d};

use b::q::{self /* After b::q::self */};
use b::r; // After b::r
use b::s::{
    a,
    b, // After b::s::b
};
use b::t::{/* Before b::t::self */ self};
use b::t::{
    // Before b::t::a
    a,
    b,
};
use b::v::{
    // Before b::v::a
    a,
    // Before b::v::b
    b,
};
use b::{
    c, d, e,
    u::{a, b},
};
use b::{
    f::g,
    h::{i, j}, /* After b::h group */
};
use b::{
    /* Before b::l group */ l::{self, m, n::o, p::*},
    q,
};
