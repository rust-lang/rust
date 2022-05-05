use std::collections::btree_map::{IterMut, OccupiedEntry, RangeMut, VacantEntry};

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn iter_cov_key<'a, 'new>(v: IterMut<'a, &'static (), ()>) -> IterMut<'a, &'new (), ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn iter_cov_val<'a, 'new>(v: IterMut<'a, (), &'static ()>) -> IterMut<'a, (), &'new ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn iter_contra_key<'a, 'new>(v: IterMut<'a, &'new (), ()>) -> IterMut<'a, &'static (), ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn iter_contra_val<'a, 'new>(v: IterMut<'a, (), &'new ()>) -> IterMut<'a, (), &'static ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}

fn range_cov_key<'a, 'new>(v: RangeMut<'a, &'static (), ()>) -> RangeMut<'a, &'new (), ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn range_cov_val<'a, 'new>(v: RangeMut<'a, (), &'static ()>) -> RangeMut<'a, (), &'new ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn range_contra_key<'a, 'new>(v: RangeMut<'a, &'new (), ()>) -> RangeMut<'a, &'static (), ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn range_contra_val<'a, 'new>(v: RangeMut<'a, (), &'new ()>) -> RangeMut<'a, (), &'static ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}

fn occ_cov_key<'a, 'new>(v: OccupiedEntry<'a, &'static (), ()>)
                         -> OccupiedEntry<'a, &'new (), ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn occ_cov_val<'a, 'new>(v: OccupiedEntry<'a, (), &'static ()>)
                         -> OccupiedEntry<'a, (), &'new ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn occ_contra_key<'a, 'new>(v: OccupiedEntry<'a, &'new (), ()>)
                            -> OccupiedEntry<'a, &'static (), ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn occ_contra_val<'a, 'new>(v: OccupiedEntry<'a, (), &'new ()>)
                            -> OccupiedEntry<'a, (), &'static ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}

fn vac_cov_key<'a, 'new>(v: VacantEntry<'a, &'static (), ()>)
                         -> VacantEntry<'a, &'new (), ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn vac_cov_val<'a, 'new>(v: VacantEntry<'a, (), &'static ()>)
                         -> VacantEntry<'a, (), &'new ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn vac_contra_key<'a, 'new>(v: VacantEntry<'a, &'new (), ()>)
                            -> VacantEntry<'a, &'static (), ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}
fn vac_contra_val<'a, 'new>(v: VacantEntry<'a, (), &'new ()>)
                            -> VacantEntry<'a, (), &'static ()> {
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ lifetime may not live long enough
}


fn main() { }
