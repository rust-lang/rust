fn main() {
    x::<#[a]y::<z>>
    //~^ ERROR attributes cannot be applied to generic arguments
    //~| ERROR cannot find value `x` in this scope
}
