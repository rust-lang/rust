//@ run-rustfix
#![allow(unreachable_code, dead_code)]

fn main() {
    fn mismatch_types1() -> i32 {
        match 1 {
            x => dbg!(x), //~ ERROR mismatched types
        }
        todo!()
    }

    fn mismatch_types2() -> i32 {
        match 2 {
            x => {
                dbg!(x) //~ ERROR mismatched types
            }
        }
        todo!()
    }

    fn mismatch_types3() -> i32 {
        match 1 {
            _ => dbg!(1) //~ ERROR mismatched types
        }
        todo!()
    }

    fn mismatch_types4() -> i32 {
        match 1 {
            _ => {dbg!(1)} //~ ERROR mismatched types
        }
        todo!()
    }
}
