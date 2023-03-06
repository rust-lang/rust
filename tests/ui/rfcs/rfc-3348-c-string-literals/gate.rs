fn main() {
    c"foo";
    //~^ ERROR: `c".."` literals are experimental

    m!(c"test");
    //~^ ERROR: `c".."` literals are experimental
}
