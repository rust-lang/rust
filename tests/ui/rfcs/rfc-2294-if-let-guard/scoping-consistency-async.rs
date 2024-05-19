// Check that temporaries in if-let guards are correctly scoped.
// Regression test for #116079.

//@ build-pass
//@ edition:2018
// -Zvalidate-mir

#![feature(if_let_guard)]

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
