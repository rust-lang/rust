// run-pass
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

struct Bencher;

// ICE
fn warm_up<'a, F>(f: F) where F: Fn(&'a mut Bencher) {
}

fn main() {
    // ICE trigger
    warm_up(|b: &mut Bencher| () );

    // OK
    warm_up(|b| () );
}
