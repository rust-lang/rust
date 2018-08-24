fn read_lines_borrowed<'a>() -> Vec<&'a str> {
    let rawLines: Vec<String> = vec!["foo  ".to_string(), "  bar".to_string()];
    rawLines //~ ERROR `rawLines` does not live long enough
        .iter().map(|l| l.trim()).collect()
}

fn main() {}
