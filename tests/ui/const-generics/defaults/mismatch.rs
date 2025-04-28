pub struct Example<const N: usize = 13>;
pub struct Example2<T = u32, const N: usize = 13>(T);
pub struct Example3<const N: usize = 13, T = u32>(T);
pub struct Example4<const N: usize = 13, const M: usize = 4>;

fn main() {
    let e: Example<13> = ();
    //~^ ERROR mismatched types
    //~| expected struct `Example`
    let e: Example2<u32, 13> = ();
    //~^ ERROR mismatched types
    //~| expected struct `Example2`
    let e: Example3<13, u32> = ();
    //~^ ERROR mismatched types
    //~| expected struct `Example3`
    let e: Example3<7> = ();
    //~^ ERROR mismatched types
    //~| expected struct `Example3<7>`
    let e: Example4<7> = ();
    //~^ ERROR mismatched types
    //~| expected struct `Example4<7>`
}
