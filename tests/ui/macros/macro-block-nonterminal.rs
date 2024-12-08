//@ run-pass

macro_rules! do_block{
    ($val:block) => {$val}
}

fn main() {
    let s;
    do_block!({ s = "it works!"; });
    assert_eq!(s, "it works!");
}
