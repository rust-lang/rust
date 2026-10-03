// Type ascription doesn't lead to unsoundness

#![feature(type_ascription)]

fn main() {
    let arr = &[1u8, 2, 3];
    let ref x = type_ascribe!(arr, &[u8]);      //~ ERROR mismatched types
    let ref mut x = type_ascribe!(arr, &[u8]);  //~ ERROR mismatched types
    match type_ascribe!(arr, &[u8]) {           //~ ERROR mismatched types
        ref x => {}
    }
    let _len = type_ascribe!(arr, &[u8]).len();              //~ ERROR mismatched types
}
