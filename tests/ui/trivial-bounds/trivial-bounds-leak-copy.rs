// Check that false Copy bounds don't leak
#![feature(trivial_bounds)]

fn copy_out_string(t: &String) -> String
where
    String: Copy,
    //~^ WARN: does not depend on any type or lifetime parameters
{
    *t
}

fn move_out_string(t: &String) -> String {
    *t //~ ERROR
}

fn main() {}
