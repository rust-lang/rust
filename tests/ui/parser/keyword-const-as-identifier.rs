fn main() {
    let const = "foo";
    //~^ ERROR expected `{`, found `=`
    //~| ERROR inline-const in pattern position is experimental [E0658]
}
