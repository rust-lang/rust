//@ check-pass
//@ aux-build:issue-57264-2.rs

extern crate issue_57264_2;

fn infer<T: issue_57264_2::PubTraitWithSingleImplementor>(arg: T) -> T { arg }

fn main() {
    infer(None).unwrap().pub_method();
}
