fn read_lines_borrowed<'a>() -> Vec<&'a str> {
    let rawLines: Vec<String> = vec!["foo  ".to_string(), "  bar".to_string()];
    rawLines //~ ERROR cannot return value referencing local variable `rawLines`
        .iter().map(|l| l.trim()).collect()
}

fn main() {}
