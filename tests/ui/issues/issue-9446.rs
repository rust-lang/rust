// run-pass
struct Wrapper(String);

impl Wrapper {
    pub fn new(wrapped: String) -> Wrapper {
        Wrapper(wrapped)
    }

    pub fn say_hi(&self) {
        let Wrapper(ref s) = *self;
        println!("hello {}", *s);
    }
}

impl Drop for Wrapper {
    fn drop(&mut self) {}
}

pub fn main() {
    {
        // This runs without complaint.
        let x = Wrapper::new("Bob".to_string());
        x.say_hi();
    }
    {
        // This fails to compile, circa 0.8-89-gc635fba.
        // error: internal compiler error: drop_ty_immediate: non-box ty
        Wrapper::new("Bob".to_string()).say_hi();
    }
}
