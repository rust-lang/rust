// This checks that the attribute validation ICE in issue #105594 doesn't
// recur.

fn main() {}

#[track_caller] //~ ERROR attribute should be applied to a function
static _A: () = ();
