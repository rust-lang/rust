fn main() {
    (|_, ()| ())([return, ()]);
    //~^ ERROR this function takes 2 arguments but 1 argument was supplied
}
