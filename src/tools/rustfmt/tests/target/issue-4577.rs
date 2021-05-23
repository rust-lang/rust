fn main() {
    let s: String = "ABAABBAA"
        .chars()
        .filter(|c| if *c == 'A' { true } else { false })
        .map(|c| -> char {
            if c == 'A' {
                '0'
            } else {
                '1'
            }
        })
        .collect();

    println!("{}", s);
}
