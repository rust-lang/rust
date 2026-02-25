#![feature(sized_hierarchy)]
trait PointeeSized {} //~ HELP this trait has no implementations, consider adding one

fn require_trait<T: PointeeSized>() {} //~ NOTE required by a bound in `require_trait`
    //~| NOTE required by this bound in `require_trait`

fn main() {
    require_trait::<i32>(); //~ ERROR the trait bound `i32: PointeeSized` is not satisfied
    //~^ NOTE the trait `PointeeSized` is not implemented for `i32`
    //~| NOTE `i32` implements similarly named trait `std::marker::PointeeSized`, but not `PointeeSized`
}
