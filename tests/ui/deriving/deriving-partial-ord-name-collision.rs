//@ check-pass
//@ edition:2021

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct NameCollision(i32);

impl NameCollision {
    pub fn cmp(&self, _: &NameCollision) {}
}

fn main() {}
