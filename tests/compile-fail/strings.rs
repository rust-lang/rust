#![feature(plugin)]
#![plugin(clippy)]

#![deny(string_add_assign)]
#![deny(string_add)]
fn main() {
    let mut x = "".to_owned();

    for _ in (1..3) {
        x = x + "."; //~ERROR you assign the result of adding something to this string.
    }
    
    let y = "".to_owned();
    let z = y + "..."; //~ERROR you add something to a string.
    
    assert_eq!(&x, &z);
}
