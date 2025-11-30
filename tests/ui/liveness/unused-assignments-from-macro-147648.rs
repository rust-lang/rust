//@ check-pass
//@ aux-build:aux_issue_147648.rs

#![deny(unused_assignments)]

extern crate aux_issue_147648;

fn main() {
    aux_issue_147648::unused_assign!(y);
}
