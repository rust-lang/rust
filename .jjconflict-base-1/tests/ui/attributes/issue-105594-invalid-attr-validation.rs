// This checks that the attribute validation ICE in issue #105594 doesn't
// recur.

fn main() {}

#[track_caller] //~ ERROR attribute cannot be used on
static _A: () = ();
