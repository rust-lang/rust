#![forbid(dead_code)]

#[derive(Debug)]
pub struct Whatever { //~ ERROR struct `Whatever` is never constructed
    pub field0: (),
    field1: (),
    field2: (),
    field3: (),
    field4: (),
}

fn main() {}
