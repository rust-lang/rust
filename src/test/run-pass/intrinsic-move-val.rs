#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn move_val_init<T>(dst: &mut T, -src: T);
    fn move_val<T>(dst: &mut T, -src: T);
}

fn main() {
    let mut x = @1;
    let mut y = @2;
    rusti::move_val(&mut y, x);
    assert *y == 1;
}