// rustfmt-file_lines: [{"file":"tests/target/issue-3442.rs","range":[5,5]},{"file":"tests/target/issue-3442.rs","range":[8,8]}]

extern crate alpha; // comment 1
extern crate beta; // comment 2
#[allow(aaa)] // comment 3
#[macro_use]
extern crate gamma;
#[allow(bbb)] // comment 4
#[macro_use]
extern crate lazy_static;
