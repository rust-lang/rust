// check-pass

enum E {}

fn f(e: E) {
    println!("{}", (e as isize).to_string());
}

fn main() {}
