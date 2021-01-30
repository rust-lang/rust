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

use {
    a::{
        aa::{aaa, aab},
        ab,
        ac::aca,
        ad::ada,
    },
    b as x,
};
