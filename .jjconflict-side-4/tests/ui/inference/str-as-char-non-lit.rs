// Don't suggest double quotes when encountering an expr of type `char` where a `&str`
// is expected if the expr is not a char literal.
// issue: rust-lang/rust#125595

fn main() {
    let _: &str = ('a'); //~ ERROR mismatched types
    let token = || 'a';
    let _: &str = token(); //~ ERROR mismatched types
}
