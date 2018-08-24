fn main() {
    let _ = Iterator::next(&mut ());
    //~^ ERROR `(): std::iter::Iterator` is not satisfied

    for _ in false {}
    //~^ ERROR `bool: std::iter::Iterator` is not satisfied

    let _ = Iterator::next(&mut ());
    //~^ ERROR `(): std::iter::Iterator` is not satisfied

    other()
}

pub fn other() {
    // check errors are still reported globally

    let _ = Iterator::next(&mut ());
    //~^ ERROR `(): std::iter::Iterator` is not satisfied

    let _ = Iterator::next(&mut ());
    //~^ ERROR `(): std::iter::Iterator` is not satisfied

    for _ in false {}
    //~^ ERROR `bool: std::iter::Iterator` is not satisfied
}
