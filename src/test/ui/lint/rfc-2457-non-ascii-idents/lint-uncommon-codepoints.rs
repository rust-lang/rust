#![feature(non_ascii_idents)]
#![deny(uncommon_codepoints)]

const µ: f64 = 0.000001; //~ ERROR identifier contains uncommon Unicode codepoints

fn dĳkstra() {} //~ ERROR identifier contains uncommon Unicode codepoints

fn main() {
    let ㇻㇲㇳ = "rust"; //~ ERROR identifier contains uncommon Unicode codepoints
    println!("{}", ㇻㇲㇳ); //~ ERROR identifier contains uncommon Unicode codepoints
}
