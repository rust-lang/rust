//@ compile-flags: -Z ui-testing=no
fn function_with_lots_of_arguments(a: i32, b: char, c: i32, d: i32, e: i32, f: i32) {}

fn main() {
    let variable_name = 42;
    function_with_lots_of_arguments(
        variable_name,
        variable_name,
        variable_name,
        variable_name,
        variable_name,
    );
    //~^^^^^^^ ERROR this function takes 6 arguments but 5 arguments were supplied [E0061]
}
