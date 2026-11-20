struct NoCloneOrEq;

#[derive(PartialEq)]
struct E {
    x: NoCloneOrEq //~ ERROR: can't compare `NoCloneOrEq`
}
#[derive(Clone)]
struct C {
    x: NoCloneOrEq
    //~^ ERROR: `NoCloneOrEq: Clone` is not satisfied
}


fn main() {}
