fn main() {
    let a = Some(vec![1, 2]);
    match a {
        Some(refe list) => println!("{list:?}"),
        //~^ ERROR expected one of
        //~| ERROR this pattern has 2 fields,
        _ => println!("none"),
    }
}
