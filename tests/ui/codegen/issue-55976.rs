// Regression test for issue #55976.

fn main() {
    type_error(|x| &x);
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

fn type_error<T>(
    _selector: for<'a> fn(&'a Vec<Box<dyn for<'b> Fn(&'b u8)>>) -> &'a Vec<Box<dyn Fn(T)>>,
) {
}
