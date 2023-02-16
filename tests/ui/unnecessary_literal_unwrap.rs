//run-rustfix
#![warn(clippy::unnecessary_literal_unwrap)]

fn unwrap_option() {
    let _val = Some(1).unwrap();
    let _val = Some(1).expect("this never happens");
}

fn unwrap_result() {
    let _val = Ok::<usize, ()>(1).unwrap();
    let _val = Ok::<usize, ()>(1).expect("this never happens");
    // let val = Err(1).unwrap_err();
}

fn main() {
    unwrap_option();
    unwrap_result();
}
