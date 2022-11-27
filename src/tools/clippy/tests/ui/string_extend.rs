// run-rustfix

#[derive(Copy, Clone)]
struct HasChars;

impl HasChars {
    fn chars(self) -> std::str::Chars<'static> {
        "HasChars".chars()
    }
}

fn main() {
    let abc = "abc";
    let def = String::from("def");
    let mut s = String::new();

    s.push_str(abc);
    s.extend(abc.chars());

    s.push_str("abc");
    s.extend("abc".chars());

    s.push_str(&def);
    s.extend(def.chars());

    s.extend(abc.chars().skip(1));
    s.extend("abc".chars().skip(1));
    s.extend(['a', 'b', 'c'].iter());

    let f = HasChars;
    s.extend(f.chars());

    // issue #9735
    s.extend(abc[0..2].chars());
}
