// check-pass

struct Wrap<T>(T);

fn match_wrap<T>(w: Wrap<T>) {
    match w {
        Wrap<T>(_) => println!("That's a wrap!"), // Look, no `::<>`!
    }
}

fn main() {}
