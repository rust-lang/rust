
fn main() {
    let i: int =
        match Some::<int>(3) { None::<int> => { fail } Some::<int>(_) => { 5 } };
    log(debug, i);
}
