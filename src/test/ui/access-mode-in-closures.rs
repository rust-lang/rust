struct sty(Vec<isize> );

fn unpack<F>(_unpack: F) where F: FnOnce(&sty) -> Vec<isize> {}

fn main() {
    let _foo = unpack(|s| {
        // Test that `s` is moved here.
        match *s { sty(v) => v } //~ ERROR cannot move out
    });
}
