fn main() {
    let x = 2;
    let x_message = alt x {
      0 to 1     { ~"not many" }
      _          { ~"lots" }
    };
    assert x_message == ~"lots";

    let y = 2i;
    let y_message = alt y {
      0 to 1     { ~"not many" }
      _          { ~"lots" }
    };
    assert y_message == ~"lots";

    let z = 1u64;
    let z_message = alt z {
      0 to 1     { ~"not many" }
      _          { ~"lots" }
    };
    assert z_message == ~"not many";
}
