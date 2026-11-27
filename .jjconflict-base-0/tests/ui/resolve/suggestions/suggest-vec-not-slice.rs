// Suggest `Vec<T>`, not `[T]` (#159491).
fn values() -> &[i32] {
    //~^ ERROR [E0106]
    //~| SUGGESTION Vec<
    //~| SUGGESTION 'static
    let values = vec![1, 2];
    &values
}
fn main(){}
