#![feature(plugin)]
#![plugin(clippy)]

// cause the build to fail if this warning is invoked
#![deny(check_for_loop_mut_bound)]

// an example
fn mut_range_bound() {
    let mut m = 4;
    for i in 0..m { continue; } // ERROR One of the range bounds is mutable
}

fn main(){
    mut_range_bound();
}
