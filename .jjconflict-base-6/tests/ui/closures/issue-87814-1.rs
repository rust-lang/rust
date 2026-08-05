//@ check-pass
fn main() {
    let mut schema_all = vec![];
    (0..42).for_each(|_x| match Err(()) as Result<(), _> {
        Ok(()) => schema_all.push(()),
        Err(_) => (),
    });
}
