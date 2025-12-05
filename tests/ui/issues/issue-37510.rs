//@ check-pass

fn foo(_: &mut i32) -> bool { true }

fn main() {
    let opt = Some(92);
    let mut x = 62;

    if let Some(_) = opt {

    } else if foo(&mut x) {

    }
}
