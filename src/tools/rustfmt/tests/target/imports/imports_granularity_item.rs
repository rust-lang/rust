// rustfmt-imports_granularity: Item

use a::b;
use a::c;
use a::d;
use a::f::g;
use a::h::i;
use a::h::j;
use a::l::m;
use a::l::n::o;
use a::l::p::*;
use a::l::{self};
use a::q::{self};

use b::c;
use b::d;
use b::e;
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
use b::u::a;
use b::u::b;
use b::v::{
    // Before b::v::a
    a,
    // Before b::v::b
    b,
};
use b::{
    f::g,
    h::{i, j}, /* After b::h group */
};
use b::{
    /* Before b::l group */ l::{self, m, n::o, p::*},
    q,
};
