enum newtype {
    newtype(int)
}

fn main() {

    // Test that borrowck treats enums with a single variant
    // specially.

    let x = @mut 5;
    let y = @mut newtype(3);
    let z = alt *y {
      newtype(b) {
        *x += 1;
        *x * b
      }
    };
    assert z == 18;
}