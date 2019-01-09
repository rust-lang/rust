struct NoCloneOrEq;

#[derive(PartialEq)]
struct E {
    x: NoCloneOrEq //~ ERROR binary operation `==` cannot be applied to type `NoCloneOrEq`
         //~^ ERROR binary operation `!=` cannot be applied to type `NoCloneOrEq`
}
#[derive(Clone)]
struct C {
    x: NoCloneOrEq
    //~^ ERROR `NoCloneOrEq: std::clone::Clone` is not satisfied
}


fn main() {}
