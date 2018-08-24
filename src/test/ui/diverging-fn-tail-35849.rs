fn assert_sizeof() -> ! {
    unsafe {
        ::std::mem::transmute::<f64, [u8; 8]>(panic!())
            //~^ ERROR mismatched types
    }
}

fn main() { }
