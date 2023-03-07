fn test1() {
    // In this test the outer 'a loop may terminate without `x` getting initialised. Although the
    // `x = loop { ... }` statement is reached, the value itself ends up never being computed and
    // thus leaving `x` uninit.
    let x: i32;
    'a: loop {
        x = loop { break 'a };
    }
    println!("{:?}", x); //~ ERROR E0381
}

// test2 and test3 should not fail.
fn test2() {
    // In this test the `'a` loop will never terminate thus making the use of `x` unreachable.
    let x: i32;
    'a: loop {
        x = loop { continue 'a };
    }
    println!("{:?}", x);
}

fn test3() {
    let x: i32;
    // Similarly, the use of variable `x` is unreachable.
    'a: loop {
        x = loop { return };
    }
    println!("{:?}", x);
}

fn main() {
}
