fn foo(b: bool) -> Result<bool,String> { //~ ERROR mismatched types
    Err("bar".to_string());
}

fn main() {
    foo(false);
}
