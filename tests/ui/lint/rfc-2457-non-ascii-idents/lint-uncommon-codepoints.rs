#![deny(uncommon_codepoints)]

const µ: f64 = 0.000001; //~ ERROR identifier contains a non normalized (NFKC) character: 'µ'
//~| WARNING should have an upper case name

fn dĳkstra() {}
//~^ ERROR identifier contains a non normalized (NFKC) character: 'ĳ'

fn main() {
    let ㇻㇲㇳ = "rust"; //~ ERROR identifier contains uncommon characters: 'ㇻ', 'ㇲ', and 'ㇳ'

    // using the same identifier the second time won't trigger the lint.
    println!("{}", ㇻㇲㇳ);
}
