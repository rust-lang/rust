fn main() {
    let foo = &16;
    //~^ HELP consider changing this to be a mutable reference
    //~| SUGGESTION &mut 16
    *foo = 32;
    //~^ ERROR cannot assign to `*foo`, which is behind a `&` reference
    let bar = foo;
    //~^ HELP consider specifying this binding's type
    *bar = 64;
    //~^ ERROR cannot assign to `*bar`, which is behind a `&` reference
}
