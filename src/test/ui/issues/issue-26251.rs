// run-pass
fn main() {
    let x = 'a';

    let y = match x {
        'a'..='b' if false => "one",
        'a' => "two",
        'a'..='b' => "three",
        _ => panic!("what?"),
    };

    assert_eq!(y, "two");
}
