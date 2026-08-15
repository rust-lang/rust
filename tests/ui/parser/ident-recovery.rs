fn ,comma() {
    //~^ ERROR expected identifier, found `,`
    struct Foo {
        x: i32,,
        //~^ ERROR expected identifier, found `,`
        y: u32,
    }
}

fn break() {
//~^ ERROR expected identifier, found keyword `break`
    let continue = 5;
    //~^ ERROR expected identifier, found keyword `continue`
}

fn main() {}
