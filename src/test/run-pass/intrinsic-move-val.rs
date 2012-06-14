#[abi = "rust-intrinsic"]
native mod rusti {
    fn move_val_init<T>(&dst: T, -src: T);
    fn move_val<T>(&dst: T, -src: T);
}

fn main() {
    let mut x = @1;
    let mut y = @2;
    rusti::move_val(y, x);
    assert *y == 1;
}