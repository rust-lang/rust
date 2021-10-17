#![feature(let_else)]

fn main() {
    let Some(_) = Some(()) else if true {
        //~^ ERROR conditional `else if` is not supported for `let...else`
        return;
    } else {
        return;
    };
}
