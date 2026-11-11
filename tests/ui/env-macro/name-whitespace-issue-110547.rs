//@ compile-flags: -C debug-assertions

fn main() {
    env!{"\t"}; //~ ERROR not defined at compile time
    env!("\t"); //~ ERROR not defined at compile time
    env!("\u{2069}"); //~ ERROR not defined at compile time
}
