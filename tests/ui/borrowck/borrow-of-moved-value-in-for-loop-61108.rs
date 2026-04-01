// https://github.com/rust-lang/rust/issues/61108
fn main() {
    let mut bad_letters = vec!['e', 't', 'o', 'i'];
    for l in bad_letters {
        // something here
    }
    bad_letters.push('s'); //~ ERROR borrow of moved value: `bad_letters`
}
