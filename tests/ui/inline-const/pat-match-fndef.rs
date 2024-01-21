#![feature(inline_const_pat)]

fn uwu() {}

fn main() {
    let x = [];
    match x[123] {
        const { uwu } => {}
        //~^ ERROR `{fn item uwu: fn()}` cannot be used in patterns
        _ => {}
    }
}
