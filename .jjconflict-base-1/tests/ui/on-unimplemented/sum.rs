// <https://github.com/rust-lang/rust/issues/105184>

fn main() {
    vec![(), ()].iter().sum::<i32>();
    //~^ ERROR

    vec![(), ()].iter().product::<i32>();
    //~^ ERROR
}
