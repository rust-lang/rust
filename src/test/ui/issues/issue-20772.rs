trait T : Iterator<Item=Self::Item>
//~^ ERROR cycle detected
//~| ERROR associated type `Item` not found for `Self`
{}

fn main() {}
