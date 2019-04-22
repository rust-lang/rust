// compile-flags: -Z borrowck=mir

// This is the third counter-example from Niko's blog post
// smallcultfollowing.com/babysteps/blog/2017/03/01/nested-method-calls-via-two-phase-borrowing/
//
// It shows that not all nested method calls on `self` are magically
// allowed by this change. In particular, a nested `&mut` borrow is
// still disallowed.

fn main() {


    let mut vec = vec![0, 1];
    vec.get({

        vec.push(2);
        //~^ ERROR cannot borrow `vec` as mutable because it is also borrowed as immutable

        0
    });
}
