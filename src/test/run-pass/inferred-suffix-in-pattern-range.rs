fn main() {
    let x = 2;
    let x_message = match x {
      0 .. 1     => { ~"not many" }
      _          => { ~"lots" }
    };
    assert x_message == ~"lots";

    let y = 2i;
    let y_message = match y {
      0 .. 1     => { ~"not many" }
      _          => { ~"lots" }
    };
    assert y_message == ~"lots";

    let z = 1u64;
    let z_message = match z {
      0 .. 1     => { ~"not many" }
      _          => { ~"lots" }
    };
    assert z_message == ~"not many";
}
