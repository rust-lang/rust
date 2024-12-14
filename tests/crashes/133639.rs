//@ known-bug: #133639

#![feature(with_negative_coherence)]
#![feature(min_specialization)]
#![feature(generic_const_exprs)]

#![crate_type = "lib"]
use std::str::FromStr;

struct a<const b: bool>;

trait c {}

impl<const d: u32> FromStr for e<d>
where
    a<{ d <= 2 }>: c,
{
    type Err = ();
    fn from_str(f: &str) -> Result<Self, Self::Err> {
        unimplemented!()
    }
}
struct e<const d: u32>;

impl<const d: u32> FromStr for e<d>
where
    a<{ d <= 2 }>: c,
{
    type Err = ();
    fn from_str(f: &str) -> Result<Self, Self::Err> {
        unimplemented!()
    }
}
