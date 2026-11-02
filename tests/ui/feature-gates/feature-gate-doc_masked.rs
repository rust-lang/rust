#[doc(masked)] //~ ERROR the `doc(masked)` attribute is experimental
extern crate std as realstd;

fn main() {
    #[doc(masked)]
    //~^ ERROR the `doc(masked)` attribute is experimental [E0658]
    println!();
}
