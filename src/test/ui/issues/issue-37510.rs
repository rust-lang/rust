#![feature(rustc_attrs)]

fn foo(_: &mut i32) -> bool { true }

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let opt = Some(92);
    let mut x = 62;

    if let Some(_) = opt {

    } else if foo(&mut x) {

    }
}
