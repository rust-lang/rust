fn main() {
    println!(
        r#"
    \"\'}､"# //~ ERROR invalid format string: unmatched `}` found
    );
}
