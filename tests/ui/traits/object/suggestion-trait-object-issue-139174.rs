//@ edition: 2021

fn f<'a>(x: Box<dyn Fn() -> Option<usize + 'a>>) -> usize {
    //~^ ERROR expected trait, found builtin type `usize`
    //~| ERROR expected a type, found a trait [E0782]
    0
}

fn create_adder<'a>(x: i32) -> usize + 'a {
    //~^ ERROR expected trait, found builtin type `usize`
    //~| ERROR expected a type, found a trait [E0782]
    move |y| x + y
}

struct Struct<'a>{
    x: usize + 'a,
    //~^ ERROR expected trait, found builtin type `usize`
    //~| ERROR expected a type, found a trait [E0782]
}


fn main() {

}
