fn test() {
    let v: int;
    loop {
        v = 1; //~ ERROR re-assignment of immutable variable
        //~^ NOTE prior assignment occurs here
        copy v; // just to prevent liveness warnings
    }
}

fn main() {
}
