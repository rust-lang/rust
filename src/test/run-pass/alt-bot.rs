
fn main() {
    let i: int =
        match some::<int>(3) { none::<int> => { fail } some::<int>(_) => { 5 } };
    log(debug, i);
}
