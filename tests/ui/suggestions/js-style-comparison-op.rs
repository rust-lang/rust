//@ run-rustfix
fn main() {
    if 1 === 1 { //~ ERROR invalid comparison operator `===`
        println!("yup!");
    } else if 1 !== 1 { //~ ERROR invalid comparison operator `!==`
        println!("nope!");
    }
}
