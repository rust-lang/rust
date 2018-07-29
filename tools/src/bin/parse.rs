extern crate libsyntax2;

use std::io::Read;

use libsyntax2::{parse_green, tokenize};
use libsyntax2::utils::dump_tree_green;

fn main() {
    let text = read_input();
    let tokens = tokenize(&text);
    let file = parse_green(text, &tokens);
    let tree = dump_tree_green(&file);
    println!("{}", tree);
}

fn read_input() -> String {
    let mut buff = String::new();
    ::std::io::stdin().read_to_string(&mut buff).unwrap();
    buff
}
