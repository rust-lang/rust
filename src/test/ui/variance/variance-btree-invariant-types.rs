use std::collections::btree_map::{IterMut, OccupiedEntry, VacantEntry};

fn iter_cov_key<'a, 'new>(v: IterMut<'a, &'static (), ()>) -> IterMut<'a, &'new (), ()> {
    v //~ ERROR mismatched types
}
fn iter_cov_val<'a, 'new>(v: IterMut<'a, (), &'static ()>) -> IterMut<'a, (), &'new ()> {
    v //~ ERROR mismatched types
}
fn iter_contra_key<'a, 'new>(v: IterMut<'a, &'new (), ()>) -> IterMut<'a, &'static (), ()> {
    v //~ ERROR mismatched types
}
fn iter_contra_val<'a, 'new>(v: IterMut<'a, (), &'new ()>) -> IterMut<'a, (), &'static ()> {
    v //~ ERROR mismatched types
}

fn occ_cov_key<'a, 'new>(v: OccupiedEntry<'a, &'static (), ()>)
                         -> OccupiedEntry<'a, &'new (), ()> {
    v //~ ERROR mismatched types
}
fn occ_cov_val<'a, 'new>(v: OccupiedEntry<'a, (), &'static ()>)
                         -> OccupiedEntry<'a, (), &'new ()> {
    v //~ ERROR mismatched types
}
fn occ_contra_key<'a, 'new>(v: OccupiedEntry<'a, &'new (), ()>)
                            -> OccupiedEntry<'a, &'static (), ()> {
    v //~ ERROR mismatched types
}
fn occ_contra_val<'a, 'new>(v: OccupiedEntry<'a, (), &'new ()>)
                            -> OccupiedEntry<'a, (), &'static ()> {
    v //~ ERROR mismatched types
}

fn vac_cov_key<'a, 'new>(v: VacantEntry<'a, &'static (), ()>)
                         -> VacantEntry<'a, &'new (), ()> {
    v //~ ERROR mismatched types
}
fn vac_cov_val<'a, 'new>(v: VacantEntry<'a, (), &'static ()>)
                         -> VacantEntry<'a, (), &'new ()> {
    v //~ ERROR mismatched types
}
fn vac_contra_key<'a, 'new>(v: VacantEntry<'a, &'new (), ()>)
                            -> VacantEntry<'a, &'static (), ()> {
    v //~ ERROR mismatched types
}
fn vac_contra_val<'a, 'new>(v: VacantEntry<'a, (), &'new ()>)
                            -> VacantEntry<'a, (), &'static ()> {
    v //~ ERROR mismatched types
}


fn main() { }
