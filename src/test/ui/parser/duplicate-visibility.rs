fn main() {}

extern "C" {
    pub pub fn foo();
    //~^ ERROR expected one of `(`, `async`, `const`, `default`, `extern`, `fn`, `pub`, `unsafe`, or `use`, found keyword `pub`
}
