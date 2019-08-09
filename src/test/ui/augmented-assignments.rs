use std::ops::AddAssign;

struct Int(i32);

impl AddAssign for Int {
    fn add_assign(&mut self, _: Int) {
        unimplemented!()
    }
}

fn main() {
    let mut x = Int(1);
    x
    //~^ NOTE borrow of `x` occurs here
    +=
    x;
    //~^ ERROR cannot move out of `x` because it is borrowed
    //~| move out of `x` occurs here

    let y = Int(2);
    //~^ HELP consider changing this to be mutable
    //~| SUGGESTION mut y
    y   //~ ERROR cannot borrow `y` as mutable, as it is not declared as mutable
        //~| cannot borrow as mutable
    +=
    Int(1);
}
