// compile-flags: -Z parse-only

fn main() {
    let x = r##"lol"#;
    //~^ ERROR unterminated raw string
}
