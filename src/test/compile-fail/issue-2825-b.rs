class example {
    let x: int;
    new() {
        self.x = 1;
    }
    drop {} //~ ERROR First destructor declared
    drop {
        debug!("Goodbye, cruel world");
    }
}

fn main(_args: ~[~str]) {
  let e: example = example();
}
