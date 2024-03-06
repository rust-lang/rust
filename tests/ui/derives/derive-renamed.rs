//@ check-pass
//@ edition:2018

use derive as my_derive;

#[my_derive(Debug)]
struct S;

fn main() {
    println!("{:?}", S); // OK
}
