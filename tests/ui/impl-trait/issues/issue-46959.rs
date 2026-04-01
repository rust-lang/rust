//@ check-pass
#![deny(non_camel_case_types)]

#[allow(dead_code)]
fn qqq(lol: impl Iterator<Item=u32>) -> impl Iterator<Item=u64> {
        lol.map(|x|x as u64)
}

fn main() {}
