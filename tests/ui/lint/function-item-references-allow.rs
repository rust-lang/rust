//@ check-pass
// regression test for https://github.com/rust-lang/rust/issues/162107

#[allow(function_item_references)]
fn main() {
    println!("{:p}", &std::env::var::<String>);
}
