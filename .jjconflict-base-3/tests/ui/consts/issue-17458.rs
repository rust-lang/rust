static X: usize = unsafe { core::ptr::null::<usize>() as usize };
//~^ ERROR: pointers cannot be cast to integers during const eval

fn main() {
    assert_eq!(X, 0);
}
