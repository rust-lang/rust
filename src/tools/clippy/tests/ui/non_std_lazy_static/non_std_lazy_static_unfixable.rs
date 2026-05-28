//@aux-build:once_cell.rs
//@aux-build:lazy_static.rs
//@no-rustfix

#![warn(clippy::non_std_lazy_statics)]
#![allow(static_mut_refs)]

mod once_cell_lazy {
    use once_cell::sync::Lazy;

    static LAZY_FOO: Lazy<String> = Lazy::new(|| "foo".to_uppercase());
    //~^ ERROR: this type has been superseded by `LazyLock` in the standard library
    static mut LAZY_BAR: Lazy<String> = Lazy::new(|| "bar".to_uppercase());
    //~^ ERROR: this type has been superseded by `LazyLock` in the standard library
    static mut LAZY_BAZ: Lazy<String> = Lazy::new(|| "baz".to_uppercase());
    //~^ ERROR: this type has been superseded by `LazyLock` in the standard library

    fn calling_irreplaceable_fns() {
        let _ = Lazy::get(&LAZY_FOO);

        unsafe {
            let _ = Lazy::get_mut(&mut LAZY_BAR);
            let _ = Lazy::force_mut(&mut LAZY_BAZ);
        }
    }
}

mod lazy_static_lazy_static {
    use lazy_static::lazy_static;

    lazy_static! {
        static ref LAZY_FOO: String = "foo".to_uppercase();
    }
    //~^^^ ERROR: this macro has been superseded by `std::sync::LazyLock`
    lazy_static! {
        static ref LAZY_BAR: String = "bar".to_uppercase();
        static ref LAZY_BAZ: String = "baz".to_uppercase();
    }
    //~^^^^ ERROR: this macro has been superseded by `std::sync::LazyLock`
    //~| ERROR: this macro has been superseded by `std::sync::LazyLock`
}

fn main() {}
