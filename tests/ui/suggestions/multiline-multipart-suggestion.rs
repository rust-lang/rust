// compile-flags: --error-format=human --color=always
// ignore-windows

fn short(foo_bar: &Vec<&i32>) -> &i32 { //~ ERROR missing lifetime specifier
    &12
}

fn long( //~ ERROR missing lifetime specifier
    foo_bar: &Vec<&i32>,
    something_very_long_so_that_the_line_will_wrap_around__________: i32,
) -> &i32 {
    &12
}

fn long2( //~ ERROR missing lifetime specifier
    foo_bar: &Vec<&i32>) -> &i32 {
    &12
}
fn main() {}
