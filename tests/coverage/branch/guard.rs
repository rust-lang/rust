#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

macro_rules! no_merge {
    () => {
        for _ in 0..1 {}
    };
}

fn branch_match_guard(x: Option<u32>) {
    no_merge!();

    match x {
        Some(0) => {
            println!("zero");
        }
        Some(x) if x % 2 == 0 => {
            println!("is nonzero and even");
        }
        Some(x) if x % 3 == 0 => {
            println!("is nonzero and odd, but divisible by 3");
        }
        _ => {
            println!("something else");
        }
    }
}

#[coverage(off)]
fn main() {
    branch_match_guard(Some(0));
    branch_match_guard(Some(2));
    branch_match_guard(Some(6));
    branch_match_guard(Some(3));
}
