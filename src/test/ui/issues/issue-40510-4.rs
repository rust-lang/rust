// build-pass (FIXME(62277): could be check-pass?)
#![allow(unused)]

fn f() {
    let x: Vec<()> = Vec::new();

    || {
        || {
            x.len()
        }
    };
}


fn main() {}
