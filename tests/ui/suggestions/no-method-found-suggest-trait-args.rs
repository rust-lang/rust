/// Tests that suggestions to add trait bounds that would enable using a method include appropriate
/// placeholder arguments for that trait.

trait Trait<I> {
    fn method(&self) {}
}

trait Trait2<'a, A, const B: u8, C = (), const D: u8 = 0> {
    fn method2(&self) {}
}

fn foo<T>(value: T) {
    //~^ SUGGESTION : Trait</* I */>
    //~| SUGGESTION : Trait2</* 'a, A, B */>
    value.method();
    //~^ ERROR no method named `method` found for type parameter `T` in the current scope [E0599]
    value.method2();
    //~^ ERROR no method named `method2` found for type parameter `T` in the current scope [E0599]
}

fn bar(value: impl Copy) {
    //~^ SUGGESTION + Trait</* I */>
    //~| SUGGESTION + Trait2</* 'a, A, B */>
    value.method();
    //~^ ERROR no method named `method` found for type parameter `impl Copy` in the current scope [E0599]
    value.method2();
    //~^ ERROR no method named `method2` found for type parameter `impl Copy` in the current scope [E0599]
}

fn main() {}
