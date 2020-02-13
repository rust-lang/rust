fn main() {}

extern {
    pub pub fn foo();
    //~^ ERROR missing `fn`, `type`, or `static` for extern-item declaration
}
