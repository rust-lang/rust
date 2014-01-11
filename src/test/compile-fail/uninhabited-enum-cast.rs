enum E {}

fn f(e: E) {
    println!("{}", (e as int).to_str());   //~ ERROR non-scalar cast
}

fn main() {}
