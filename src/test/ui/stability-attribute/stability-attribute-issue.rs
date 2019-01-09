// aux-build:stability_attribute_issue.rs
// ignore-tidy-linelength

#![deny(deprecated)]

extern crate stability_attribute_issue;
use stability_attribute_issue::*;

fn main() {
    unstable();
    //~^ ERROR use of unstable library feature 'unstable_test_feature' (see issue #1)
    unstable_msg();
    //~^ ERROR use of unstable library feature 'unstable_test_feature': message (see issue #2)
}
