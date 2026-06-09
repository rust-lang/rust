// Check that temporaries in if-let guards are correctly scoped.
// Regression test for #116079.

//@ check-pass
//@revisions: edition2021 edition2024 edition2018
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@[edition2018] edition:2018
//@ compile-flags: -Zvalidate-mir

static mut A: [i32; 5] = [1, 2, 3, 4, 5];

async fn fun() {
    unsafe {
        match A {
            _ => (),
            i if let Some(1) = async { Some(1) }.await => (),
            _ => (),
        }
    }
}

async fn funner() {
    unsafe {
        match A {
            _ => (),
            _ | _ if let Some(1) = async { Some(1) }.await => (),
            _ => (),
        }
    }
}

fn main() {}
