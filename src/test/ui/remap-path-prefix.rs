// compile-flags: --remap-path-prefix={{src-base}}=remapped

fn main() {
    // We cannot actually put an ERROR marker here because
    // the file name in the error message is not what the
    // test framework expects (since the filename gets remapped).
    // We still test the expected error in the stderr file.
    ferris
}
