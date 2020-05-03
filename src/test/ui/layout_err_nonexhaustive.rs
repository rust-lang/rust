use std::mem::LayoutError;

fn main() {
    let _err = LayoutError; //~ ERROR expected value, found struct `LayoutError`
}
