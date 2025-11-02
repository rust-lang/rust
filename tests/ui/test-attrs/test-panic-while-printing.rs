//@ compile-flags:--test
//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc

use std::fmt;
use std::fmt::{Display, Formatter};

pub struct A(Vec<u32>);

impl Display for A {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result {
        self.0[0];
        Ok(())
    }
}

#[test]
fn main() {
    let result = std::panic::catch_unwind(|| {
        let a = A(vec![]);
        eprintln!("{}", a);
    });
    assert!(result.is_err());
}
