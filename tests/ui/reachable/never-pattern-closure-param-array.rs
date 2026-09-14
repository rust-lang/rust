//@ check-pass
//@ edition: 2024

#![feature(never_patterns)]
#![allow(incomplete_features)]
#![allow(unreachable_code)]

fn main() {
    let _ = Some({
        return;
    })
    .map(|!| [1]);
}
