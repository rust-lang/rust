#![feature(ascii_char)] // random lib feature
#![feature(box_patterns)] // random lang feature

// picked arbitrary unstable features, just need a random lib and lang feature, ideally ones that
// won't be stabilized any time soon so we don't have to update this test
fn main() {
    for s in quix("foo/bar") {
        print!("{s}");
    }
    println!();
}

// need a latebound var to trigger the incremental compilation ICE
fn quix(foo: &str) -> impl Iterator<Item = &'_ str> + '_ {
    foo.split('/')
}
