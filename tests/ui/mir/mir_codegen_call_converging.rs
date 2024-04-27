//@ run-pass
fn converging_fn() -> u64 {
    43
}

fn mir() -> u64 {
    let x;
    loop {
        x = converging_fn();
        break;
    }
    x
}

fn main() {
    assert_eq!(mir(), 43);
}
