// Regression test for #92105.
// ICE when mutating immutable reference from last statement of a block.

fn main() {
    let foo = Some(&0).unwrap();
    *foo = 1; //~ ERROR cannot assign
}
