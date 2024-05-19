fn main() {}

struct Type;

impl Type {
    undef!();
    //~^ ERROR cannot find macro `undef` in this scope
}
