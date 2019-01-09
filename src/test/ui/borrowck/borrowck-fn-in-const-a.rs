// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

// Check that we check fns appearing in constant declarations.
// Issue #22382.

const MOVE: fn(&String) -> String = {
    fn broken(x: &String) -> String {
        return *x //[ast]~ ERROR cannot move out of borrowed content [E0507]
                  //[mir]~^ ERROR [E0507]
    }
    broken
};

fn main() {
}
