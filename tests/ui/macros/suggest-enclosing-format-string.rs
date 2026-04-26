// Suggest enclosing the format string with `""` when it is one of `{}`, `{:?}`, and `{:#?}`.

#[derive(Debug)]
enum UwU {
    QwQ,
    AwA,
    QAQ,
}

fn main() {
    println!({}, UwU::QwQ);
    //~^ ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    //~| HELP you might want to enclose `{}` with `""`
    println!({:?}, UwU::QwQ);
    //~^ ERROR expected expression, found `:`
    //~| ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    //~| HELP maybe write a path separator here
    //~| HELP you might want to enclose `{:?}` with `""`
    println!({:#?}, UwU::QwQ);
    //~^ ERROR expected expression, found `:`
    //~| ERROR format argument must be a string literal
    //~| HELP you might be missing a string literal to format with
    //~| HELP maybe write a path separator here
    //~| HELP you might want to enclose `{:#?}` with `""`
}
