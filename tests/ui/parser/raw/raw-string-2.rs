fn main() {
    let x = r###"here's a long string"# "# "##;
    //~^ ERROR unterminated raw string
}
