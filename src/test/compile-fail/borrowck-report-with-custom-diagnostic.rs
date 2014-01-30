#[allow(dead_code)];
fn main() {
    // Original borrow ends at end of function
    let mut x = 1u;
    let y = &mut x;
    let z = &x; //~ ERROR cannot borrow
}
//~^ NOTE previous borrow ends here

fn foo() {
    match true {
        true => {
            // Original borrow ends at end of match arm
            let mut x = 1u;
            let y = &x;
            let z = &mut x; //~ ERROR cannot borrow
        }
     //~^ NOTE previous borrow ends here
        false => ()
    }
}

fn bar() {
    // Original borrow ends at end of closure
    || {
        let mut x = 1u;
        let y = &mut x;
        let z = &mut x; //~ ERROR cannot borrow
    };
 //~^ NOTE previous borrow ends here
}
