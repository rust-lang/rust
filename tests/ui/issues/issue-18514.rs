// run-pass
// Test that we don't ICE when codegenning a generic impl method from
// an extern crate that contains a match expression on a local
// variable place where one of the match case bodies contains an
// expression that autoderefs through an overloaded generic deref
// impl.

// aux-build:issue-18514.rs

extern crate issue_18514 as ice;
use ice::{Tr, St};

fn main() {
    let st: St<()> = St(vec![]);
    st.tr();
}
