#![deny(instantly_dangling_pointer)]

const MAX_PATH: usize = 260;
fn main() {
    let str1 = String::with_capacity(MAX_PATH).as_mut_ptr();
    //~^ ERROR [instantly_dangling_pointer]
    let str2 = String::from("TotototototototototototototototototoT").as_ptr();
    //~^ ERROR [instantly_dangling_pointer]
    unsafe {
        std::ptr::copy_nonoverlapping(str2, str1, 30);
        println!("{:?}", String::from_raw_parts(str1, 30, 30));
    }
}
