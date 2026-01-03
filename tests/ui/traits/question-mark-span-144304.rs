//@ ignore-arm - armhf-gnu have more types implement trait `From<T>`, let's skip it
fn f() -> Result<(), i32> {
    Err("str").map_err(|e| e)?; //~ ERROR `?` couldn't convert the error to `i32`
    Err("str").map_err(|e| e.to_string())?; //~ ERROR `?` couldn't convert the error to `i32`
    Ok(())
}

fn main() {}
