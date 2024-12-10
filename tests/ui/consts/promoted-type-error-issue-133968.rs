struct B<T: ?Sized + Send + 'static> {
    x: &'static T,
}
static STR: &'static [u8] = "a b"; //~ERROR: mismatched types
static C: &B<[u8]> = &B { x: STR };

fn main() {}
