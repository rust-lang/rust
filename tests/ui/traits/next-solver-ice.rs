//@compile-flags: -Znext-solver=globally
//@check-fail

fn check<T: Iterator>() {
    <f32 as From<<T as Iterator>::Item>>::from;
    //~^ ERROR the trait bound `f32: From<<T as Iterator>::Item>` is not satisfied
}

fn main() {}
