//@ edition:2018

#![deny(unused_imports)]

mod multi_segment {
    use core::any; //~ ERROR unused import: `core::any`
}

mod single_segment {
    use core; //~ ERROR unused import: `core`
}

mod single_segment_underscore {
    use core as _; // OK
}

fn main() {}
