#![forbid(dead_code)]

#[derive(Debug)]
pub struct Whatever {
    pub field0: (),
    field1: (), //~ERROR field is never read: `field1
    field2: (), //~ERROR field is never read: `field2
    field3: (), //~ERROR field is never read: `field3
    field4: (), //~ERROR field is never read: `field4
}

fn main() {}
