fn main() {
    x::<#[a]y::<z>>
    //~^ ERROR attributes cannot be applied to generic type arguments
    //~| ERROR cannot find value `x` in this scope
}
