// rustfmt-imports_granularity: One

use b;
use a::ac::{aca, acb};
use a::{aa::*, ab};

use a as x;
use b::ba;
use a::{aa, ab};

use a::aa::aaa;
use a::ab::aba as x;
use a::aa::*;

use a::aa;
use a::ad::ada;
#[cfg(test)]
use a::{ab, ac::aca};
use b;
#[cfg(test)]
use b::{
    ba, bb,
    bc::bca::{bcaa, bcab},
};

pub use a::aa;
pub use a::ae;
use a::{ab, ac, ad};
use b::ba;
pub use b::{bb, bc::bca};

use a::aa::aaa;
use a::ac::{aca, acb};
use a::{aa::*, ab};
use b::{
    ba,
    bb::{self, bba},
};

use crate::a;
use crate::b::ba;
use c::ca;

use super::a;
use c::ca;
use super::b::ba;

use crate::a;
use super::b;
use c::{self, ca};

use a::{
    // some comment
    aa::{aaa, aab},
    ab,
    // another comment
    ac::aca,
};
use b as x;
use a::ad::ada;

use b::{f::g, h::{i, j} /* After b::h group */};
use b::e;
use b::{/* Before b::l group */ l::{self, m, n::o, p::*}, q};
use b::d;
use b::r; // After b::r
use b::q::{self /* After b::q::self */};
use b::u::{
    a,
    b,
};
use b::t::{
    // Before b::t::a
    a,
    b,
};
use b::s::{
    a,
    b, // After b::s::b
};
use b::v::{
    // Before b::v::a
    a,
    // Before b::v::b
    b,
};
use b::t::{/* Before b::t::self */ self};
use b::c;
