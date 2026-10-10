struct Demo {
    val: u32,
}

macro_rules! as_ptr {
    ($d:expr) => {
        &mut $d as *mut Demo
    };
}

macro_rules! get_value {
    ($d:expr) => {
        as_ptr!($d).val
        //~^ ERROR no field `val` on type `*mut Demo`
    }
}

fn main() {
    let mut d = Demo { val: 123 };
    let _ = get_value!(d);
}
