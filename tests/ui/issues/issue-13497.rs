fn read_lines_borrowed1() -> Vec<
    &str //~ ERROR missing lifetime specifier
> {
    let rawLines: Vec<String> = vec!["foo  ".to_string(), "  bar".to_string()];
    rawLines.iter().map(|l| l.trim()).collect()
}

fn main() {}
