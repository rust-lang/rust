// issue #89807



#[deny(unused_variables)]

fn main() {
    let value = Some(String::new());
    #[allow(unused)]
    let banana = 1;
    #[allow(unused)]
    let Some(chaenomeles) = value.clone() else { return }; // OK

    let Some(chaenomeles) = value else { return }; //~ ERROR unused variable: `chaenomeles`
}
