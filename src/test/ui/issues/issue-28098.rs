fn main() {
    let _ = Iterator::next(&mut ());
    //~^ ERROR `()` is not an iterator

    for _ in false {}
    //~^ ERROR `bool` is not an iterator

    let _ = Iterator::next(&mut ());
    //~^ ERROR `()` is not an iterator

    other()
}

pub fn other() {
    // check errors are still reported globally

    let _ = Iterator::next(&mut ());
    //~^ ERROR `()` is not an iterator

    let _ = Iterator::next(&mut ());
    //~^ ERROR `()` is not an iterator

    for _ in false {}
    //~^ ERROR `bool` is not an iterator
}
