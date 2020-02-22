fn main() {}

extern {
    pub pub fn foo();
    //~^ ERROR missing `fn`, `type`, `const`, or `static` for item declaration
}
