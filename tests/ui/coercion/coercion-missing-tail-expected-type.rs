// #41425 -- error message "mismatched types" has wrong types
//@ run-rustfix

fn plus_one(x: i32) -> i32 { //~ ERROR mismatched types
    x + 1;
}

fn foo() -> Result<u8, u64> { //~ ERROR mismatched types
    Ok(1);
}

fn main() {
    let x = plus_one(5);
    let _ = foo();
    println!("X = {}", x);
}
