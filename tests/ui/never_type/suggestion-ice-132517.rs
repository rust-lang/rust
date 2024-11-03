fn main() {
    x::<_>(|_| panic!())
    //~^ ERROR cannot find function `x` in this scope
}
