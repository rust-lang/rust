fn main() {
    let _result = &Some(42).as_deref();
//~^ ERROR the method
}
