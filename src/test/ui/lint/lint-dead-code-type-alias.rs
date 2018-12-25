#![deny(dead_code)]

type Used = u8;
type Unused = u8; //~ ERROR type alias is never used

fn id(x: Used) -> Used { x }

fn main() {
    id(0);
}
