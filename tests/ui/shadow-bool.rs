//@ check-pass

mod bar {
    pub trait QueryId {
        const SOME_PROPERTY: bool;
    }
}

use bar::QueryId;

#[allow(non_camel_case_types)]
pub struct bool;

impl QueryId for bool {
    const SOME_PROPERTY: core::primitive::bool = true;
}

fn main() {}
