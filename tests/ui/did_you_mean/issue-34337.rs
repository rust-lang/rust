//@ check-pass
fn get(key: &mut String) { }

fn main() {
    let mut v: Vec<String> = Vec::new();
    let ref mut key = v[0];
    get(&mut key); //~ WARNING cannot borrow
    //~| HELP try removing `&mut` here
}
