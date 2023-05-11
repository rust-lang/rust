fn main() {
    let _result = &Ok(42).as_deref();
//~^ ERROR the method
}
