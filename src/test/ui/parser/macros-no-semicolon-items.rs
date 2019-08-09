macro_rules! foo()  //~ ERROR semicolon
                    //~| ERROR unexpected end of macro

macro_rules! bar {
    ($($tokens:tt)*) => {}
}

bar!( //~ ERROR semicolon
    blah
    blah
    blah
)

fn main() {
}
