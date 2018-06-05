fn main() {
    fn f() {}

    let g = unsafe {
        std::mem::transmute::<fn(), fn(i32)>(f)
    };

    g(42) //~ ERROR constant evaluation error [E0080]
    //~^ NOTE tried to call a function with sig fn() through a function pointer of type fn(i32)
}
