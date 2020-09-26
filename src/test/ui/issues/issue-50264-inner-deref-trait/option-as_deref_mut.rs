fn main() {
    let _result = &mut Some(42).as_deref_mut();
//~^ ERROR the method
}
