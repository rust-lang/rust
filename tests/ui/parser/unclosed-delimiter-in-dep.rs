//@ ignore-backends: gcc

mod unclosed_delim_mod;

fn main() {
    let _: usize = unclosed_delim_mod::new();
}

//~? ERROR mismatched closing delimiter: `}`
