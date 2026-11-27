//! Regression test for <https://github.com/rust-lang/rust/issues/3847>.
//! Pattern matching struct from different module caused parser to fail.
//@ run-pass

mod buildings {
    pub struct Tower { pub height: usize }
}

pub fn main() {
    let sears = buildings::Tower { height: 1451 };
    let h: usize = match sears {
        buildings::Tower { height: h } => { h }
    };

    println!("{}", h);
}
