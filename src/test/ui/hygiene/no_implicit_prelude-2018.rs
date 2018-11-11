// edition:2018

#[no_implicit_prelude]
mod bar {
    fn f() {
        ::std::print!(""); // OK
        print!(); //~ ERROR cannot find macro `print!` in this scope
    }
}

fn main() {}
