fn main() {
    (|_, ()| ())(if true {} else {return;});
    //~^ ERROR function takes 2 arguments but 1 argument was supplied
}
