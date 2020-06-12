fn id<T>(t: T) -> T { t }
fn main() {
    const A: bool = unsafe { id::<u8> as *const () < id::<u16> as *const () };
    //~^ ERROR pointers cannot be compared in a meaningful way during const eval
    println!("{}", A);
}
