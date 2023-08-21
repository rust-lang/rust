#![feature(inline_const_pat)]
//~^ WARN the feature `inline_const_pat` is incomplete

fn uwu() {}

fn main() {
    let x = [];
    match x[123] {
        const { uwu } => {}
        //~^ ERROR `fn() {uwu}` cannot be used in patterns
        _ => {}
    }
}
