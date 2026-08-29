// rustfmt-imports_granularity: One

use {
    a::{
        aa::*,
        ab,
        ac::{aca, acb},
    },
    b,
};

use {
    a::{self as x, aa, ab},
    b::ba,
};

use a::{
    aa::{aaa, *},
    ab::aba as x,
};

#[cfg(test)]
use a::{ab, ac::aca};
#[cfg(test)]
use b::{
    ba, bb,
    bc::bca::{bcaa, bcab},
};
use {
    a::{aa, ad::ada},
    b,
};

pub use {
    a::{aa, ae},
    b::{bb, bc::bca},
};
use {
    a::{ab, ac, ad},
    b::ba,
};

use {
    a::{
        aa::{aaa, *},
        ab,
        ac::{aca, acb},
    },
    b::{
        ba,
        bb::{self, bba},
    },
};

use {
    crate::{a, b::ba},
    c::ca,
};

use {
    super::{a, b::ba},
    c::ca,
};

use {
    super::b,
    crate::a,
    c::{self, ca},
};

use a::{
    // some comment
    aa::{aaa, aab},
    ab,
    // another comment
    ac::aca,
};
use {a::ad::ada, b as x};

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
