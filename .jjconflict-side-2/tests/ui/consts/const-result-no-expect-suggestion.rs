const fn f(value: u32) -> Result<u32, ()> {
    Ok(value)
}

const TEST: u32 = f(2);
//~^ ERROR: mismatched types

const fn g() -> Result<String, ()> {
    Ok(String::new())
}

const TEST2: usize = g().len();
//~^ ERROR: no method named `len` found for enum `Result<T, E>`

fn main() {}
