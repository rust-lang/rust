enum thing = uint;
impl thing : cmp::Ord { //~ ERROR missing method `gt`
    pure fn lt(&&other: thing) -> bool { *self < *other }
    pure fn le(&&other: thing) -> bool { *self < *other }
    pure fn ge(&&other: thing) -> bool { *self < *other }
}
fn main() {}
