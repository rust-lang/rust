use std::ops::AddAssign;

struct Int(i32);

impl AddAssign for Int {
    fn add_assign(&mut self, _: Int) {
        unimplemented!()
    }
}

fn main() {
    let mut x = Int(1);
    x   //~ error: use of moved value: `x`
    //~^ value used here after move
    +=
    x;  //~ value moved here

    let y = Int(2);
    //~^ HELP make this binding mutable
    //~| SUGGESTION mut y
    y   //~ error: cannot borrow immutable local variable `y` as mutable
        //~| cannot borrow
    +=
    Int(1);
}
