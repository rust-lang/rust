// aux-build:issue-42708.rs

#![feature(decl_macro)]
#![allow(unused)]

extern crate issue_42708;

macro m() {
    #[derive(issue_42708::Test)]
    struct S { x: () }

    #[issue_42708::attr_test]
    struct S2 { x: () }

    #[derive(Clone)]
    struct S3 { x: () }

    fn g(s: S, s2: S2, s3: S3) {
        (s.x, s2.x, s3.x);
    }
}

m!();

fn main() {}
