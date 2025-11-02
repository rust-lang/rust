fn outlives_indir<'a: 'b, 'b, T: 'a>(_x: T) {}
//~^ NOTE: requirements that the value outlives `'b` introduced here

fn foo<'b>() { //~ NOTE: lifetime `'b` defined here
    outlives_indir::<'_, 'b, _>(&mut 1u32); //~ ERROR: temporary value dropped while borrowed
    //~^ NOTE: argument requires that borrow lasts for `'b`
    //~| NOTE: creates a temporary value which is freed while still in use
    //~| NOTE: temporary value is freed at the end of this statement
}

fn main() {}
