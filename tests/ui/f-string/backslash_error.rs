pub fn main() {
    let a = "foo\{"; //~ ERROR unknown character escape: `{`
    let b = "bar\}"; //~ ERROR unknown character escape: `}`
}
