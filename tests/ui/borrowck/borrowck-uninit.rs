fn foo(x: isize) { println!("{}", x); }

fn main() {
    let x: isize;
    foo(x); //~ ERROR E0381
}
