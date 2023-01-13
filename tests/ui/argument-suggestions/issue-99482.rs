fn main() {
    let f = |_: (), f: fn()| f;
    let _f = f(main);
    //~^ ERROR function takes 2 arguments but 1 argument was supplied
}
