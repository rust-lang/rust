struct example {
    x: int,
    drop {} //~ ERROR First destructor declared
    drop {
        debug!("Goodbye, cruel world");
    }
}

fn example() -> example {
    example {
        x: 1
    }
}

fn main(_args: ~[~str]) {
  let e: example = example();
}
