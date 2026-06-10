trait Super<T> {
    type Assoc;
}

trait Sub: Super<i32, Assoc = u32> + Super<i64, Assoc = u64> {}

fn foo(_: &dyn Sub) {}
//~^ ERROR conflicting associated type bindings for `Assoc`

fn main() {}
