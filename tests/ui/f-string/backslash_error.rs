pub fn main() {
    let a = "foo\{";
    //~ ERROR invalid escape
    let b = "bar\}";
    //~ ERROR invalid escape
}
