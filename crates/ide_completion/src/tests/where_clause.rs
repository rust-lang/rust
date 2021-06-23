//! Completion tests for inside of where clauses.
//!
//! The parent of the where clause tends to bleed completions of itself into the where clause so this
//! has to be thoroughly tested.
use expect_test::{expect, Expect};

use crate::tests::completion_list;

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}
