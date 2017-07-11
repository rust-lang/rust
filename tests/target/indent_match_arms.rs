// rustfmt-indent_match_arms: false

fn main() {
    match x {
    1 => "one",
    2 => "two",
    3 => "three",
    4 => "four",
    5 => "five",
    _ => "something else",
    }

    match x {
    1 => "one",
    2 => "two",
    3 => "three",
    4 => "four",
    5 => match y {
    'a' => 'A',
    'b' => 'B',
    'c' => 'C',
    _ => "Nope",
    },
    _ => "something else",
    }

}
