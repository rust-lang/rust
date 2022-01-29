trait T : Iterator<Item=Self::Item>
//~^ ERROR cycle detected
{}

fn main() {}
