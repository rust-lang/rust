fn main() {}

extern "C" {
    default!(); //~ ERROR cannot find macro `default` in this scope
    default do
    //~^ ERROR unmatched `default`
    //~| ERROR non-item in item list
}
