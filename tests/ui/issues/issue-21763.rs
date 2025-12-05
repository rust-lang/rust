// Regression test for HashMap only impl'ing Send/Sync if its contents do

//@ normalize-stderr: "\S+[\\/]hashbrown\S+" -> "$$HASHBROWN_SRC_LOCATION"

use std::collections::HashMap;
use std::rc::Rc;

fn foo<T: Send>() {}

fn main() {
    foo::<HashMap<Rc<()>, Rc<()>>>();
    //~^ ERROR `Rc<()>` cannot be sent between threads safely
}
