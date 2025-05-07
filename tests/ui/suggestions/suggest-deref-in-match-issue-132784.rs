use std::sync::Arc;
fn main() {
    let mut x = Arc::new(Some(1));
    match x {
        //~^ HELP consider dereferencing to access the inner value using the Deref trait
        //~| HELP consider dereferencing to access the inner value using the Deref trait
        Some(_) => {}
        //~^ ERROR mismatched types
        None => {}
        //~^ ERROR mismatched types
    }

    match &x {
        //~^ HELP consider dereferencing to access the inner value using the Deref trait
        //~| HELP consider dereferencing to access the inner value using the Deref trait
        Some(_) => {}
        //~^ ERROR mismatched types
        None => {}
        //~^ ERROR mismatched types
    }

    let mut y = Box::new(Some(1));
    match y {
        //~^ HELP consider dereferencing to access the inner value using the Deref trait
        //~| HELP consider dereferencing to access the inner value using the Deref trait
        Some(_) => {}
        //~^ ERROR mismatched types
        None => {}
        //~^ ERROR mismatched types
    }

    let mut z = Arc::new(Some(1));
    match z as Arc<Option<i32>> {
        //~^ HELP consider dereferencing to access the inner value using the Deref trait
        //~| HELP consider dereferencing to access the inner value using the Deref trait
        Some(_) => {}
        //~^ ERROR mismatched types
        None => {}
        //~^ ERROR mismatched types
    }

    let z_const: &Arc<Option<i32>> = &z;
    match z_const {
        //~^ HELP consider dereferencing to access the inner value using the Deref trait
        //~| HELP consider dereferencing to access the inner value using the Deref trait
        Some(_) => {}
        //~^ ERROR mismatched types
        None => {}
        //~^ ERROR mismatched types
    }

    // Normal reference because Arc doesn't implement DerefMut.
    let z_mut: &mut Arc<Option<i32>> = &mut z;
    match z_mut {
        //~^ HELP consider dereferencing to access the inner value using the Deref trait
        //~| HELP consider dereferencing to access the inner value using the Deref trait
        Some(_) => {}
        //~^ ERROR mismatched types
        None => {}
        //~^ ERROR mismatched types
    }

    // Mutable reference because Box does implement DerefMut.
    let y_mut: &mut Box<Option<i32>> = &mut y;
    match y_mut {
        //~^ HELP consider dereferencing to access the inner value using the Deref trait
        //~| HELP consider dereferencing to access the inner value using the Deref trait
        Some(_) => {}
        //~^ ERROR mismatched types
        None => {}
        //~^ ERROR mismatched types
    }

    // Difficult expression.
    let difficult = Arc::new(Some(1));
    match (& (&difficult)  ) {
        //~^ HELP consider dereferencing to access the inner value using the Deref trait
        //~| HELP consider dereferencing to access the inner value using the Deref trait
        Some(_) => {}
        //~^ ERROR mismatched types
        None => {}
        //~^ ERROR mismatched types
    }
}
