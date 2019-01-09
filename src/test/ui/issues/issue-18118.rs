pub fn main() {
    const z: &'static isize = {
        //~^ ERROR let bindings in constants are unstable
        //~| ERROR statements in constants are unstable
        let p = 3;
        //~^ ERROR let bindings in constants are unstable
        //~| ERROR statements in constants are unstable
        &p //~ ERROR `p` does not live long enough
        //~^ ERROR let bindings in constants are unstable
    };
}
