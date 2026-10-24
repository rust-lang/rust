//@ edition:2018

macro_rules! r#await {
    () => { println!("Hello, world!") }
}

fn main() {
    await!()
    //~^ ERROR expected expression, found `)`
}
