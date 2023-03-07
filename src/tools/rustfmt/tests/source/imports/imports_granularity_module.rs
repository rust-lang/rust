// rustfmt-imports_granularity: Module

use a::{b::c, d::e};
use a::{f, g::{h, i}};
use a::{j::{self, k::{self, l}, m}, n::{o::p, q}};
pub use a::{r::s, t};
use b::{c::d, self};

#[cfg(test)]
use foo::{a::b, c::d};
use foo::e;

use bar::{
    // comment
    a::b,
    // more comment
    c::d,
    e::f,
};

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
