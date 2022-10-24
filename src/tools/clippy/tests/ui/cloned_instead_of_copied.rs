// run-rustfix

#![feature(custom_inner_attributes)]
#![warn(clippy::cloned_instead_of_copied)]
#![allow(unused)]

fn main() {
    // yay
    let _ = [1].iter().cloned();
    let _ = vec!["hi"].iter().cloned();
    let _ = Some(&1).cloned();
    let _ = Box::new([1].iter()).cloned();
    let _ = Box::new(Some(&1)).cloned();

    // nay
    let _ = [String::new()].iter().cloned();
    let _ = Some(&String::new()).cloned();
}

fn msrv_1_34() {
    #![clippy::msrv = "1.34"]

    let _ = [1].iter().cloned();
    let _ = Some(&1).cloned();
}

fn msrv_1_35() {
    #![clippy::msrv = "1.35"]

    let _ = [1].iter().cloned();
    let _ = Some(&1).cloned(); // Option::copied needs 1.35
}

fn msrv_1_36() {
    #![clippy::msrv = "1.36"]

    let _ = [1].iter().cloned(); // Iterator::copied needs 1.36
    let _ = Some(&1).cloned();
}
