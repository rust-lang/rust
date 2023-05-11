// Regression test for #66270, fixed by #66246

struct Bug {
    incorrect_field: 0,
    //~^ ERROR expected type
}

struct Empty {}

fn main() {
    let Bug {
        any_field: Empty {},
    } = Bug {};
}
