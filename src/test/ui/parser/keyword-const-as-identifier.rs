fn main() {
    let const = "foo";
    //~^ ERROR expected `{`, found `=`
    //~| ERROR inline-const is experimental [E0658]
}
