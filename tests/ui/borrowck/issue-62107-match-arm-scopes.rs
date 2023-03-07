fn main() {
    let e: i32;
    match e {
        //~^ ERROR E0381
        ref u if true => {}
        ref v if true => {
            let tx = 0;
            &tx;
        }
        _ => (),
    }
}
