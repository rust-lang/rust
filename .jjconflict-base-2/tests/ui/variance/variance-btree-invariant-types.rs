use std::collections::btree_map::{IterMut, OccupiedEntry, RangeMut, VacantEntry};

fn iter_cov_key<'a, 'new>(v: IterMut<'a, &'static (), ()>) -> IterMut<'a, &'new (), ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn iter_cov_val<'a, 'new>(v: IterMut<'a, (), &'static ()>) -> IterMut<'a, (), &'new ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn iter_contra_key<'a, 'new>(v: IterMut<'a, &'new (), ()>) -> IterMut<'a, &'static (), ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn iter_contra_val<'a, 'new>(v: IterMut<'a, (), &'new ()>) -> IterMut<'a, (), &'static ()> {
    v
    //~^ ERROR lifetime may not live long enough
}

fn range_cov_key<'a, 'new>(v: RangeMut<'a, &'static (), ()>) -> RangeMut<'a, &'new (), ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn range_cov_val<'a, 'new>(v: RangeMut<'a, (), &'static ()>) -> RangeMut<'a, (), &'new ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn range_contra_key<'a, 'new>(v: RangeMut<'a, &'new (), ()>) -> RangeMut<'a, &'static (), ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn range_contra_val<'a, 'new>(v: RangeMut<'a, (), &'new ()>) -> RangeMut<'a, (), &'static ()> {
    v
    //~^ ERROR lifetime may not live long enough
}

fn occ_cov_key<'a, 'new>(v: OccupiedEntry<'a, &'static (), ()>)
                         -> OccupiedEntry<'a, &'new (), ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn occ_cov_val<'a, 'new>(v: OccupiedEntry<'a, (), &'static ()>)
                         -> OccupiedEntry<'a, (), &'new ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn occ_contra_key<'a, 'new>(v: OccupiedEntry<'a, &'new (), ()>)
                            -> OccupiedEntry<'a, &'static (), ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn occ_contra_val<'a, 'new>(v: OccupiedEntry<'a, (), &'new ()>)
                            -> OccupiedEntry<'a, (), &'static ()> {
    v
    //~^ ERROR lifetime may not live long enough
}

fn vac_cov_key<'a, 'new>(v: VacantEntry<'a, &'static (), ()>)
                         -> VacantEntry<'a, &'new (), ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn vac_cov_val<'a, 'new>(v: VacantEntry<'a, (), &'static ()>)
                         -> VacantEntry<'a, (), &'new ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn vac_contra_key<'a, 'new>(v: VacantEntry<'a, &'new (), ()>)
                            -> VacantEntry<'a, &'static (), ()> {
    v
    //~^ ERROR lifetime may not live long enough
}
fn vac_contra_val<'a, 'new>(v: VacantEntry<'a, (), &'new ()>)
                            -> VacantEntry<'a, (), &'static ()> {
    v
    //~^ ERROR lifetime may not live long enough
}


fn main() { }
