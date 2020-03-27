extern {
    fn cc_plus_one_c(arg : &u32) -> u32;
    fn cc_plus_one_c_asm(arg : &u32) -> u32;
    fn cc_plus_one_cxx(arg : &u32) -> u32;
    fn cc_plus_one_cxx_asm(arg : &u32) -> u32;
}

fn main() {
    let value : u32 = 41;
    
    unsafe{
        println!("Answer to the Ultimate Question of Life, the Universe, and Everything: {}!", cc_plus_one_c(&value));
        println!("Answer to the Ultimate Question of Life, the Universe, and Everything: {}!", cc_plus_one_c_asm(&value));
        println!("Answer to the Ultimate Question of Life, the Universe, and Everything: {}!", cc_plus_one_cxx(&value));
        println!("Answer to the Ultimate Question of Life, the Universe, and Everything: {}!", cc_plus_one_cxx_asm(&value));
    }
}
