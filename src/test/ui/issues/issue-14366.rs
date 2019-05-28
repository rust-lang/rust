fn main() {
    let _x = "test" as &dyn (::std::any::Any);
    //~^ ERROR the size for values of type
}
