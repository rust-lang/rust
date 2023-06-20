// check-pass
// aux-build:issue-112831-aux.rs

mod zeroable {
    pub trait Zeroable {}
}

use zeroable::*;

mod pod {
    use super::*;
    pub trait Pod: Zeroable {}
}

use pod::*;

extern crate issue_112831_aux;
use issue_112831_aux::Zeroable;

fn main() {}
