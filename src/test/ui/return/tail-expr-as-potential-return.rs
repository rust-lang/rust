fn main() {
    let _ = foo(true);
}

fn foo(x: bool) -> Result<f64, i32> {
    if x {
        Err(42) //~ ERROR mismatched types
    }
    Ok(42.0)
}
