// test that an Index projection fails after a sibling ConstantIndex projection is moved out of
// regression test for #160525

fn main() {
    let mut arr = [[Box::new(42)]];
    let alias = &mut arr[0][{ let [row] = arr; drop(row); 0 }]; //~ ERROR
    println!("{}", **alias); // use-after-free of arr's dead stack slot
}
