// compile-flags: --edition=2021

fn main() {
    let cond = std::env::args().len() == 1;
    if cond {
        println!("true");
    }
    println!("done");
}
