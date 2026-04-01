// Regression test for one part of issue #105306.

fn main() {
    let _ = Option::<[u8]>::None;
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}
