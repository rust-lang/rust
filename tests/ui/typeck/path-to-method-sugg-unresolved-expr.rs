fn main() {
    let page_size = page_size::get();
    //~^ ERROR cannot find item `page_size`
    //~| NOTE use of undeclared crate or module `page_size`
}
