fn main() {
    x::<#[a]y::<z>>
    //~^ ERROR invalid const generic expression
    //~| ERROR cannot find value `x` in this scope
}
