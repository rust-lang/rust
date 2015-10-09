struct A;

fn main() {
    let a = A;

    if a == a {} //~ ERROR binary operation `==` cannot be applied to type `A`
        //^~ NOTE an implementation of `std::cmp::PartialEq` might be missing for `A` or one of

    if a < a {} //~ ERROR binary operation `<` cannot be applied to type `A`
        //^~ NOTE an implementation of `std::cmp::PartialOrd` might be missing for `A` or one of

    if a <= a {} //~ ERROR binary operation `<=` cannot be applied to type `A`
        //^~ NOTE an implementation of `std::cmp::PartialOrd` might be missing for `A` or one of

    if a > a {} //~ ERROR binary operation `>` cannot be applied to type `A`
        //^~ NOTE an implementation of `std::cmp::PartialOrd` might be missing for `A` or one of

    if a >= a {} //~ ERROR binary operation `>=` cannot be applied to type `A`
        //^~ NOTE an implementation of `std::cmp::PartialOrd` might be missing for `A` or one of
}
