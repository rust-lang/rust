//@ revisions: edition2015 edition2018 edition2021
//@ [edition2015] edition:2015
//@ [edition2018] edition:2018
//@ [edition2021] edition:2021
//@ compile-flags: --extern import_alias_issue_121168_extern
//@ aux-build: import-alias-issue-121168-extern.rs

extern crate import_alias_issue_121168_extern as nice_crate_name;

fn use_foo_from_another_crate_without_importing_it_first() {
    let _: Foo<i32> = todo!(); //~ ERROR cannot find type `Foo` in this scope
}

fn main() {}
