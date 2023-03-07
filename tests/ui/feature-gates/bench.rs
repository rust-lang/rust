// edition:2018

#[bench] //~ ERROR use of unstable library feature 'test'
         //~| WARN this was previously accepted
fn bench() {}

use bench as _; //~ ERROR use of unstable library feature 'test'
                //~| WARN this was previously accepted
fn main() {}
