// see #9186

enum Bar<T> { What } //~ ERROR parameter `T` is never used

fn foo<T>() {
    static a: Bar<T> = Bar::What;
//~^ ERROR can't use generic parameters from outer function
}

fn main() {
}
