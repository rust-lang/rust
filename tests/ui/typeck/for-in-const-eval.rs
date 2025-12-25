//! regression test for issue <https://github.com/rust-lang/rust/issues/50582>
fn main() {
    Vec::<[(); 1 + for x in 0..1 {}]>::new();
    //~^ ERROR cannot add
}
