fn main() {
    let e: i32;
    match e {
        ref u if true => {}
        //~^ ERROR E0381
        ref v if true => {
            let tx = 0;
            &tx;
        }
        _ => (),
    }
}
