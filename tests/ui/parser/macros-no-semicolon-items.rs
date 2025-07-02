macro_rules! foo()  //~ ERROR semicolon
                    //~| ERROR macros must contain at least one rule

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
