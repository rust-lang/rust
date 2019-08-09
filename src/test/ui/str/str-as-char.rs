// run-rustfix

fn main() {
    println!('●●'); //~ ERROR character literal may only contain one codepoint
}
