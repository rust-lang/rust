fn read_lines_borrowed<'a>() -> Vec<&'a str> {
    let raw_lines: Vec<String> = vec!["foo  ".to_string(), "  bar".to_string()];
    raw_lines.iter().map(|l| l.trim()).collect()
    //~^ ERROR `raw_lines` does not live long enough
}

fn main() {
    println!("{:?}", read_lines_borrowed());
}
