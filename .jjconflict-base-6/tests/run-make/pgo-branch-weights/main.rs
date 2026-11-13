extern crate interesting;

fn main() {
    let arg = std::env::args().skip(1).next().unwrap();

    for c in arg.chars() {
        if c == '2' {
            interesting::function_called_twice(c);
        } else {
            interesting::function_called_42_times(c);
        }

        if c == '0' {
            interesting::function_called_never(c);
        }
    }
}
