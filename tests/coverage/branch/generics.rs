#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

fn print_size<T>() {
    if std::mem::size_of::<T>() > 4 {
        println!("size > 4");
    } else {
        println!("size <= 4");
    }
}

#[coverage(off)]
fn main() {
    print_size::<()>();
    print_size::<u32>();
    print_size::<u64>();
}
