extern "C" {
    static TEST1: i32;
    fn test1(i: i32);
}

unsafe extern "C" {
    //~^ ERROR: extern block cannot be declared unsafe
    static TEST2: i32;
    fn test2(i: i32);
}

fn main() {}
