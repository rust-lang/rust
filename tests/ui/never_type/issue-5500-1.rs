// the test is here to check that the compiler doesn't ICE (cf. #5500).

struct TrieMapIterator<'a> {
    node: &'a usize,
}

fn main() {
    let a = 5;
    let _iter = TrieMapIterator { node: &a };
    _iter.node = &panic!()
    //~^ ERROR mismatched types
    //~| expected `&usize`, found `&!`
    //~| expected reference `&usize`
    //~| found reference `&!`
}
