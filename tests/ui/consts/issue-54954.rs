const ARR_LEN: usize = Tt::const_val::<[i8; 123]>();
//~^ ERROR E0790

trait Tt {
    const fn const_val<T: Sized>() -> usize {
        //~^ ERROR functions in traits cannot be declared const
        core::mem::size_of::<T>()
    }
}

fn f(z: [f32; ARR_LEN]) -> [f32; ARR_LEN] {
    z
}

fn main() {
    let _ = f([1f32; ARR_LEN]);
}
