//@ edition: 2024
#![allow(incomplete_features)]
#![feature(ref_pat_eat_one_layer_2024)]

pub fn main() {
    if let Some(&Some(x)) = Some(&Some(&mut 0)) {
        //~^ ERROR: cannot move out of a shared reference [E0507]
        let _: &u32 = x;
    }

    let &ref mut x = &0;
    //~^ cannot borrow data in a `&` reference as mutable [E0596]
}
