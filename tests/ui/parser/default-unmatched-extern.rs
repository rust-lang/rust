fn main() {}

extern "C" {
    default!(); //~ ERROR cannot find macro `default`
    default do
    //~^ ERROR `default` is not followed by an item
    //~| ERROR non-item in item list
}
