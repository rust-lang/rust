extern {
    fn cc_plus_one_c(arg : &u32) -> u32;
}

fn main() {
    let value : u32 = 41;
    
    unsafe{
        println!("Answer to the Ultimate Question of Life, the Universe, and Everything: {}!", cc_plus_one_c(&value));
    }
}
