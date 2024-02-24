//@ edition:2018

#![no_core]
#![feature(no_core)]

// @matchesraw 'issue_89852/sidebar-items.js' '"repro"'
// @!matchesraw 'issue_89852/sidebar-items.js' '"repro".*"repro"'

#[macro_export]
macro_rules! repro {
    () => {};
}

pub use crate::repro as repro2;
