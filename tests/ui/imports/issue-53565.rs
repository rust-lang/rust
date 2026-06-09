use std::time::{foo, bar, buzz};
//~^ ERROR unresolved imports
use std::time::{abc, def};
//~^ ERROR unresolved imports
fn main(){
    println!("Hello World!");
}
