fn bar(int_param: int) {}

fn main() {
     let foo: [u8, ..4] = [1u8, ..4u8];
     bar(foo); //~ ERROR mismatched types: expected `int` but found `[u8, .. 4]` (expected int but found vector)
}
