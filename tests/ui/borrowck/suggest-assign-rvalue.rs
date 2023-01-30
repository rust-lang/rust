#![allow(dead_code)]
#![feature(never_type)]

#[derive(Debug, Default)]
struct Demo {}

#[derive(Debug)]
struct DemoNoDef {}

fn apple(_: u32) {}

fn banana() {
    let chaenomeles;
    apple(chaenomeles);
    //~^ ERROR used binding `chaenomeles` isn't initialized [E0381]
}

fn main() {
    let my_bool: bool = bool::default();
    println!("my_bool: {}", my_bool);

    let my_float: f32;
    println!("my_float: {}", my_float);
    //~^ ERROR used binding `my_float` isn't initialized
    let demo: Demo;
    println!("demo: {:?}", demo);
    //~^ ERROR used binding `demo` isn't initialized

    let demo_no: DemoNoDef;
    println!("demo_no: {:?}", demo_no);
    //~^ ERROR used binding `demo_no` isn't initialized

    let arr: [i32; 5];
    println!("arr: {:?}", arr);
    //~^ ERROR used binding `arr` isn't initialized
    let foo: Vec<&str>;
    println!("foo: {:?}", foo);
    //~^ ERROR used binding `foo` isn't initialized

    let my_string: String;
    println!("my_string: {}", my_string);
    //~^ ERROR used binding `my_string` isn't initialized

    let my_int: &i32;
    println!("my_int: {}", *my_int);
    //~^ ERROR used binding `my_int` isn't initialized

    let hello: &str;
    println!("hello: {}", hello);
    //~^ ERROR used binding `hello` isn't initialized

    let never: !;
    println!("never: {}", never);
    //~^ ERROR used binding `never` isn't initialized [E0381]

    banana();
}
