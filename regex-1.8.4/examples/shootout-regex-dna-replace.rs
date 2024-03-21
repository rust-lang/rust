use std::io::{self, Read};

macro_rules! regex {
    ($re:expr) => {{
        use regex::internal::ExecBuilder;
        ExecBuilder::new($re).build().unwrap().into_regex()
    }};
}

fn main() {
    let mut seq = String::with_capacity(50 * (1 << 20));
    io::stdin().read_to_string(&mut seq).unwrap();
    let ilen = seq.len();

    seq = regex!(">[^\n]*\n|\n").replace_all(&seq, "").into_owned();
    println!("original: {}, replaced: {}", ilen, seq.len());
}
