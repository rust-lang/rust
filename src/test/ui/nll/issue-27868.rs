// Regression test for issue #27868

use std::ops::AddAssign;

struct MyVec<T>(Vec<T>);

impl <T> Drop for MyVec<T> {
    fn drop(&mut self) {
        println!("Being dropped.");
    }
}

impl<T> AddAssign<T> for MyVec<T> {
    fn add_assign(&mut self, _elem: T) {
        println!("In add_assign.");
    }
}

fn main() {
    let mut vec = MyVec(vec![0]);
    let mut vecvec = vec![vec];

    vecvec[0] += {
        vecvec = vec![];
        //~^ ERROR cannot assign to `vecvec` because it is borrowed [E0506]
        0
    };
}
