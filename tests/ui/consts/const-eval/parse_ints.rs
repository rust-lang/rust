const _OK: () = match i32::from_str_radix("-1234", 10) {
    Ok(x) => assert!(x == -1234),
    Err(_) => panic!(),
};
const _TOO_LOW: () = { u64::from_str_radix("12345ABCD", 1); }; //~ ERROR evaluation of constant value failed
const _TOO_HIGH: () = { u64::from_str_radix("12345ABCD", 37); }; //~ ERROR evaluation of constant value failed

fn main () {}
