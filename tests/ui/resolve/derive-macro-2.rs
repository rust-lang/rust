//@ check-pass
//@ proc-macro: issue-112831-aux.rs

extern crate issue_112831_aux;
use issue_112831_aux::Zeroable;

mod z {
    pub trait Zeroable {}
}

use z::*;

mod pod {
    use super::*;
    pub trait Pod: Zeroable {}
}

fn main() {}
