fn foo<T>(x: T) {
    fn bar(f: Box<dyn FnMut(T) -> T>) { }
    //~^ ERROR can't use generic parameters from outer item
    //~| ERROR can't use generic parameters from outer item
}
fn main() { foo(1); }
