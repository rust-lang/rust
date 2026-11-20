//@ check-fail
fn main() {
    let x = Box::new(Some(1));

    let test: Option<i32> = x;
    //~^ ERROR mismatched types
    let x = Box::new(Some(1));
    let test: Option<i32> = { x as Box<Option<i32>> };
    //~^ ERROR mismatched types

    let x = Box::new(Some(1));
    let test: Option<i32> = if true { x as Box<Option<i32>> } else { None };
    //~^ ERROR mismatched types

    let x = std::rc::Rc::new(Some(1));
    let test: Option<i32> = x as std::rc::Rc<Option<i32>>;
    //~^ ERROR mismatched types
}
