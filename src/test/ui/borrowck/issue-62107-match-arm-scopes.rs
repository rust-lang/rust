fn main() {
    let e: i32;
    match e {
        //~^ ERROR use of possibly-uninitialized variable
        ref u if true => {}
        ref v if true => {
            let tx = 0;
            &tx;
        }
        _ => (),
    }
}
