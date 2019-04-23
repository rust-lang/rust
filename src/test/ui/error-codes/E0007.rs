fn main() {
    let x = Some("s".to_string());
    match x {
        op_string @ Some(s) => {},
        //~^ ERROR E0007
        //~| ERROR E0303
        //~| ERROR E0382
        None => {},
    }
}
