static X: usize = 5;

#[allow(mutable_transmutes)]
fn main() {
    let _x = unsafe {
        std::mem::transmute::<&usize, &mut usize>(&X) //~ ERROR borrow stack
    };
}
