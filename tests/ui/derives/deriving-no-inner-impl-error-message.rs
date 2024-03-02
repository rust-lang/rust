struct NoCloneOrEq;

#[derive(PartialEq)]
struct E {
    x: NoCloneOrEq //~ ERROR binary operation `==` cannot be applied to type `NoCloneOrEq`
}
#[derive(Clone)]
struct C {
    x: NoCloneOrEq
    //~^ ERROR trait `Clone` is not implemented for `NoCloneOrEq`
}


fn main() {}
