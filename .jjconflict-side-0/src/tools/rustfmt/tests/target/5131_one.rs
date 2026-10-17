// rustfmt-imports_granularity: One

pub use foo::{x, x as x2, y};
use {
    bar::{
        a,
        b::{self, f, g},
        c,
        d::{e, e as e2},
    },
    qux::{h, i},
};
