// Ensure that we reject code when a nonlocal exit (`break`,
// `continue`) causes us to pop over a needed assignment.

pub fn main() {
    foo1();
    foo2();
}

pub fn foo1() {
    let x: i32;
    loop { x = break; }
    println!("{}", x); //~ ERROR borrow of possibly-uninitialized variable: `x`
}

pub fn foo2() {
    let x: i32;
    for _ in 0..10 { x = continue; }
    println!("{}", x); //~ ERROR borrow of possibly-uninitialized variable: `x`
}
