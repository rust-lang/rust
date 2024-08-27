//! This is a small example using slice::chunks, which creates a very large Tree Borrows tree.
//! Thanks to ##3837, the GC now compacts the tree, so this test can be run in a reasonable time again.
//! The actual code is adapted from tiny_skia, see https://github.com/RazrFalcon/tiny-skia/blob/master/src/pixmap.rs#L121
//! To make this benchmark demonstrate the effectiveness, run with MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-provenance-gc=100"

const N: usize = 1000;

fn input_vec() -> Vec<u8> {
    vec![0; N]
}

fn main() {
    let data_len = 2 * N;
    let mut rgba_data = Vec::with_capacity(data_len);
    let img_data = input_vec();
    for slice in img_data.chunks(2) {
        let gray = slice[0];
        let alpha = slice[1];
        rgba_data.push(gray);
        rgba_data.push(gray);
        rgba_data.push(gray);
        rgba_data.push(alpha);
    }

    assert_eq!(rgba_data.len(), data_len);
}
