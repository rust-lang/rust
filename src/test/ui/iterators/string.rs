fn main() {
    for _ in "".to_owned() {}
    //~^ ERROR `std::string::String` is not an iterator
    for _ in "" {}
    //~^ ERROR `&str` is not an iterator
}
