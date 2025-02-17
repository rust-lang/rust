fn main() {
    dbg!("Is this \“ a small sized quote or a big sized quote. \“ ");
    //~^ 2:20: 2:21: unknown character escape: `\u{201c}`
    //~^^ 2:65: 2:66: unknown character escape: `\u{201c}`
}
