//@ dont-require-annotations: NOTE

use std::fs::File;
use std::sync::Arc;
use std::rc::Rc;

fn main(){
    let file = File::open("foo.txt").unwrap();
    (file, file);
    //~^ ERROR use of moved value: `file`
    //~| NOTE you can use `File::try_clone` to duplicate a `File` instance

    let arc = Arc::new(42);
    //~^ NOTE this move could be avoided by cloning the original `Arc`, which is inexpensive
    (arc, arc);
    //~^ ERROR the type `Arc` does not implement `Copy`
    //~| NOTE consider using `Arc::clone`


    let rc = Rc::new(12);
    //~^ NOTE this move could be avoided by cloning the original `Rc`, which is inexpensive
    (rc, rc);
    //~^ ERROR the type `Rc` does not implement `Copy`
    //~| NOTE consider using `Rc::clone`
}
