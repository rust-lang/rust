//@ check-pass

#![feature(adt_const_params, unsized_const_params, generic_const_exprs)]
//~^ WARN the feature `unsized_const_params` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]
//~^^ WARN the feature `generic_const_exprs` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

pub struct Changes<const CHANGES: &'static [&'static str]>
where
    [(); CHANGES.len()]:,
{
    changes: [usize; CHANGES.len()],
}

impl<const CHANGES: &'static [&'static str]> Changes<CHANGES>
where
    [(); CHANGES.len()]:,
{
    pub fn combine(&mut self, other: &Self) {
        for _change in &self.changes {}
    }
}

pub fn main() {}
