// E0277 should point exclusively at line 14, not the entire for loop span

fn main() {
    for c in "asdf" {
    //~^ ERROR `&str` is not an iterator
    //~| NOTE `&str` is not an iterator
    //~| HELP the trait `std::iter::Iterator` is not implemented for `&str`
    //~| NOTE required by `std::iter::IntoIterator::into_iter`
        println!("");
    }
}
