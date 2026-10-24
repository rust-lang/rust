fn assert_static<T: 'static>(_t: T) {}

fn main() {
    let line = String::new();
    match [&*line] { //~ ERROR `line` does not live long enough
        [ word ] => { assert_static(word); }
    }
}
