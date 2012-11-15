enum thing = uint;
impl thing : cmp::Ord { //~ ERROR missing method `gt`
    pure fn lt(&self, other: &thing) -> bool { **self < **other }
    pure fn le(&self, other: &thing) -> bool { **self < **other }
    pure fn ge(&self, other: &thing) -> bool { **self < **other }
}
fn main() {}
