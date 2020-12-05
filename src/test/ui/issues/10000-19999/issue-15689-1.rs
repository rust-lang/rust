// run-pass

#[derive(PartialEq, Debug)]
enum Test<'a> {
    Slice(&'a isize)
}

fn main() {
    assert_eq!(Test::Slice(&1), Test::Slice(&1))
}
