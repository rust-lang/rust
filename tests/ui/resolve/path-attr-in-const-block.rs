// issue#126516
// issue#126647

fn main() {
    const {
        #![path = foo!()]
        //~^ ERROR: cannot find macro `foo` in this scope
        //~| ERROR malformed `path` attribute input
    }
}
