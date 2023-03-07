#![forbid(dead_code)]

#[derive(Debug)]
pub struct Whatever {
    pub field0: (),
    field1: (), //~ ERROR fields `field1`, `field2`, `field3`, and `field4` are never read
    field2: (),
    field3: (),
    field4: (),
}

fn main() {}
