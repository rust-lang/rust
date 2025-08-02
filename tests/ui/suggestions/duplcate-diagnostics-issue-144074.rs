struct Struct<T>(T);
fn assert_sized(_x: impl Sized) {}
fn f() {
    assert_sized(Struct(*"")); //~ ERROR: the size for values of type `str` cannot be known at compilation time
    //~^ ERROR: the size for values of type `str` cannot be known at compilation time
}

fn main() {
    f();
}
