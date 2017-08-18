// rustfmt-match_pattern_separator_break_point: Front
// Whether `|` goes to front or to back.

fn main() {
    match lorem {
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        | bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        | cccccccccccccccccccccccccccccccccccccccc
        | dddddddddddddddddddddddddddddddddddddddd => (),
        _ => (),
    }
}
