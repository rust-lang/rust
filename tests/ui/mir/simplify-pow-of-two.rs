// run-pass

fn slow_2_u(a: u32) -> u32 {
    2u32.pow(a)
}

fn slow_2_i(a: u32) -> i32 {
    2i32.pow(a)
}

fn slow_4_u(a: u32) -> u32 {
    4u32.pow(a)
}

fn slow_4_i(a: u32) -> i32 {
    4i32.pow(a)
}

fn slow_256_u(a: u32) -> u32 {
    256u32.pow(a)
}

fn slow_256_i(a: u32) -> i32 {
    256i32.pow(a)
}

fn main() {
    assert_eq!(slow_2_u(0), 1);
    assert_eq!(slow_2_i(0), 1);
    assert_eq!(slow_2_u(1), 2);
    assert_eq!(slow_2_i(1), 2);
    assert_eq!(slow_2_u(2), 4);
    assert_eq!(slow_2_i(2), 4);
    assert_eq!(slow_4_u(4), 256);
    assert_eq!(slow_4_i(4), 256);
    assert_eq!(slow_4_u(15), 1073741824);
    assert_eq!(slow_4_i(15), 1073741824);
    assert_eq!(slow_4_u(16), 0);
    assert_eq!(slow_4_i(16), 0);
    assert_eq!(slow_4_u(17), 0);
    assert_eq!(slow_4_i(17), 0);
    assert_eq!(slow_256_u(2), 65536);
    assert_eq!(slow_256_i(2), 65536);
}
