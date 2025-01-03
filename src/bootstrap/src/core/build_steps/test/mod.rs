//! Build-and-run steps for `./x.py test` test fixtures
//!
//! `./x.py test` (aka [`Kind::Test`]) is currently allowed to reach build steps in other modules.
//! However, this contains ~all test parts we expect people to be able to build and run locally.

mod book_tests;
mod compiler_crate_tests;
mod compiletest_test_suites;
mod crate_tests;
mod miri_tests;
mod shared;
mod tool_based_tests;

pub(crate) use book_tests::*;
pub(crate) use compiler_crate_tests::*;
pub(crate) use compiletest_test_suites::*;
pub(crate) use crate_tests::*;
pub(crate) use miri_tests::*;
pub(crate) use tool_based_tests::*;
