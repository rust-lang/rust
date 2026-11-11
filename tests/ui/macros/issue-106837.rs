fn main() {
    concat!(-42);
    concat!(-3.14);

    concat!(-"hello");
    //~^ ERROR expected a literal

    concat!(--1);
    //~^ ERROR expected a literal
}
