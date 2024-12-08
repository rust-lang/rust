#![warn(clippy::manual_pattern_char_comparison)]

struct NotStr;

impl NotStr {
    fn find(&self, _: impl FnMut(char) -> bool) {}
}

fn main() {
    let sentence = "Hello, world!";
    sentence.trim_end_matches(|c: char| c == '.' || c == ',' || c == '!' || c == '?');
    sentence.split(|c: char| c == '\n' || c == 'X');
    sentence.split(|c| c == '\n' || c == 'X');
    sentence.splitn(3, |c: char| c == 'X');
    sentence.splitn(3, |c: char| c.is_whitespace() || c == 'X');
    let char_compare = 'X';
    sentence.splitn(3, |c: char| c == char_compare);
    sentence.split(|c: char| matches!(c, '\n' | 'X' | 'Y'));
    sentence.splitn(3, |c: char| matches!(c, 'X'));
    sentence.splitn(3, |c: char| matches!(c, 'X' | 'W'));
    sentence.find(|c| c == '🎈');

    let not_str = NotStr;
    not_str.find(|c: char| c == 'X');

    "".find(|c| c == 'a' || c > 'z');

    let x = true;
    "".find(|c| c == 'a' || x || c == 'b');

    let d = 'd';
    "".find(|c| c == 'a' || d == 'b');

    "".find(|c| match c {
        'a' | 'b' => true,
        _ => c.is_ascii(),
    });

    "".find(|c| matches!(c, 'a' | 'b' if false));

    "".find(|c| matches!(c, 'a' | '1'..'4'));
    "".find(|c| c == 'a' || matches!(c, '1'..'4'));
    macro_rules! m {
        ($e:expr) => {
            $e == '?'
        };
    }
    "".find(|c| m!(c));
}

#[clippy::msrv = "1.57"]
fn msrv_1_57() {
    let sentence = "Hello, world!";
    sentence.trim_end_matches(|c: char| c == '.' || c == ',' || c == '!' || c == '?');
}

#[clippy::msrv = "1.58"]
fn msrv_1_58() {
    let sentence = "Hello, world!";
    sentence.trim_end_matches(|c: char| c == '.' || c == ',' || c == '!' || c == '?');
}
