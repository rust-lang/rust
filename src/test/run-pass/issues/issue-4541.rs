// run-pass
// ignore-cloudabi no std::env

fn parse_args() -> String {
    let args: Vec<_> = ::std::env::args().collect();
    let mut n = 0;

    while n < args.len() {
        match &*args[n] {
            "-v" => (),
            s => {
                return s.to_string();
            }
        }
        n += 1;
    }

    return "".to_string()
}

pub fn main() {
    println!("{}", parse_args());
}
