// edition:2018

#[no_implicit_prelude]
//~^ WARNING: deprecated
mod bar {
    fn f() {
        ::std::print!(""); // OK
        print!(); //~ ERROR cannot find macro `print` in this scope
    }
}

fn main() {}
