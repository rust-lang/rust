fn main() {
    fn foo(_: Option()) {} //~ ERROR parenthesized type parameters may only be used with a `Fn` trait [E0214]
    //~^ ERROR enum takes 1 generic argument but 0 generic arguments were supplied [E0107]
    //~| HELP add missing generic argument
}
