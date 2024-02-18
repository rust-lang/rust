//@ build-pass
//@ edition:2018

#![feature(if_let_guard)]

static A: [i32; 5] = [1, 2, 3, 4, 5];

async fn fun() {
    let u = A[async { 1 }.await];
    match A {
        i if async { true }.await => (),
        i if let Some(1) = async { Some(1) }.await => (),
        _ => (),
    }
}

fn main() {
    async {
        let u = A[async { 1 }.await];
    };
    async {
        match A {
            i if async { true }.await => (),
            i if let Some(2) = async { Some(2) }.await => (),
            _ => (),
        }
    };
}
