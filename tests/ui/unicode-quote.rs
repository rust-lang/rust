// Provide a useful error message when attempting to escape a non-ascii quotation mark.
// <https://github.com/rust-lang/rust/issues/128858>
fn main() {
    dbg!("Is this \“ a small sized quote or a big sized quote. \“ ");
    //~^ ERROR 4:20: 4:21: unknown character escape: `\u{201c}`
    //~^^ ERROR 4:65: 4:66: unknown character escape: `\u{201c}`
}
