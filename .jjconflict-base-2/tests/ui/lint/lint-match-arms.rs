fn deny_on_arm() {
    match 0 {
        #[deny(unused_variables)]
        //~^ NOTE the lint level is defined here
        y => (),
        //~^ ERROR unused variable
    }
}

#[deny(unused_variables)]
fn allow_on_arm() {
    match 0 {
        #[allow(unused_variables)]
        y => (), // OK
    }
}

fn main() {}
