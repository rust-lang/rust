extern crate libsyntax2;

use std::io::Read;

use libsyntax2::{parse};
use libsyntax2::utils::dump_tree_green;

fn main() {
    let text = read_input();
    let file = parse(text);
    let tree = dump_tree_green(&file);
    println!("{}", tree);
}

fn read_input() -> String {
    let mut buff = String::new();
    ::std::io::stdin().read_to_string(&mut buff).unwrap();
    buff
}
