fn main() {
    (|_, ()| ())(if true {} else {return;});
    //~^ ERROR this function takes 2 arguments but 1 argument was supplied
}
