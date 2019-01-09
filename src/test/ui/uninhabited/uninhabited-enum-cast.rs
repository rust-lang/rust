enum E {}

fn f(e: E) {
    println!("{}", (e as isize).to_string());   //~ ERROR non-primitive cast
}

fn main() {}
