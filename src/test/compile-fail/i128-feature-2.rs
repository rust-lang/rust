fn test1() -> i128 { //~ ERROR 128-bit type is unstable
    0
}

fn test1_2() -> u128 { //~ ERROR 128-bit type is unstable
    0
}

fn test3() {
    let x: i128 = 0; //~ ERROR 128-bit type is unstable
}

fn test3_2() {
    let x: u128 = 0; //~ ERROR 128-bit type is unstable
}

#[repr(u128)]
enum A { //~ ERROR 128-bit type is unstable
    A(u64)
}
