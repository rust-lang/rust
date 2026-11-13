//! Tests that all lifetime parameters in struct (`S`) and enum (`E`) constructors are
//! treated as early bound, similar to associated items, rather than late bound as in manual
//! constructors.

struct S<'a, 'b>(&'a u8, &'b u8);
enum E<'a, 'b> {
    V(&'a u8),
    U(&'b u8),
}

fn main() {
    S(&0, &0); // OK
    S::<'static>(&0, &0);
    //~^ ERROR struct takes 2 lifetime arguments
    S::<'static, 'static, 'static>(&0, &0);
    //~^ ERROR struct takes 2 lifetime arguments
    E::V(&0); // OK
    E::V::<'static>(&0);
    //~^ ERROR enum takes 2 lifetime arguments
    E::V::<'static, 'static, 'static>(&0);
    //~^ ERROR enum takes 2 lifetime arguments
}
