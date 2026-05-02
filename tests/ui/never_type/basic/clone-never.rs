//! regression test for issue #2151

fn main() {
    let x = panic!(); //~ ERROR type annotations needed
    x.clone();
}
