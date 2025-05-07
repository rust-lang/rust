#![deny(dangling_pointers_from_temporaries)]

const MAX_PATH: usize = 260;
fn main() {
    let str1 = String::with_capacity(MAX_PATH).as_mut_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped
    let str2 = String::from("TotototototototototototototototototoT").as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `String` will be dropped
    unsafe {
        std::ptr::copy_nonoverlapping(str2, str1, 30);
        println!("{:?}", String::from_raw_parts(str1, 30, 30));
    }
}
