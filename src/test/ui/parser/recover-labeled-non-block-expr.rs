fn main() {
    'label: 1 + 1; //~ ERROR expected `while`, `for`, `loop` or `{` after a label

    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
