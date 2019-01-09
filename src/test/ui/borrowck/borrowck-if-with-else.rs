fn foo(x: isize) { println!("{}", x); }

fn main() {
    let x: isize;
    if 1 > 2 {
        println!("whoops");
    } else {
        x = 10;
    }
    foo(x); //~ ERROR use of possibly uninitialized variable: `x`
}
