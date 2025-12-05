//@aux-build:once_cell.rs
//@aux-build:lazy_static.rs

#![warn(clippy::non_std_lazy_statics)]
#![allow(static_mut_refs)]

use once_cell::sync::Lazy;

fn main() {}

static LAZY_FOO: Lazy<String> = Lazy::new(|| "foo".to_uppercase());
//~^ ERROR: this type has been superseded by `LazyLock` in the standard library
static LAZY_BAR: Lazy<String> = Lazy::new(|| {
    //~^ ERROR: this type has been superseded by `LazyLock` in the standard library
    let x = "bar";
    x.to_uppercase()
});
static LAZY_BAZ: Lazy<String> = { Lazy::new(|| "baz".to_uppercase()) };
//~^ ERROR: this type has been superseded by `LazyLock` in the standard library
static LAZY_QUX: Lazy<String> = {
    //~^ ERROR: this type has been superseded by `LazyLock` in the standard library
    if "qux".len() == 3 {
        Lazy::new(|| "qux".to_uppercase())
    } else if "qux".is_ascii() {
        Lazy::new(|| "qux".to_lowercase())
    } else {
        Lazy::new(|| "qux".to_string())
    }
};

fn non_static() {
    let _: Lazy<i32> = Lazy::new(|| 1);
    let _: Lazy<String> = Lazy::new(|| String::from("hello"));
    #[allow(clippy::declare_interior_mutable_const)]
    const DONT_DO_THIS: Lazy<i32> = Lazy::new(|| 1);
}

mod once_cell_lazy_with_fns {
    use once_cell::sync::Lazy;

    static LAZY_FOO: Lazy<String> = Lazy::new(|| "foo".to_uppercase());
    //~^ ERROR: this type has been superseded by `LazyLock` in the standard library
    static LAZY_BAR: Lazy<String> = Lazy::new(|| "bar".to_uppercase());
    //~^ ERROR: this type has been superseded by `LazyLock` in the standard library
    static mut LAZY_BAZ: Lazy<String> = Lazy::new(|| "baz".to_uppercase());
    //~^ ERROR: this type has been superseded by `LazyLock` in the standard library

    fn calling_replaceable_fns() {
        let _ = Lazy::force(&LAZY_FOO);
        let _ = Lazy::force(&LAZY_BAR);
        unsafe {
            let _ = Lazy::force(&LAZY_BAZ);
        }
    }
}

#[clippy::msrv = "1.79"]
mod msrv_not_meet {
    use lazy_static::lazy_static;
    use once_cell::sync::Lazy;

    static LAZY_FOO: Lazy<String> = Lazy::new(|| "foo".to_uppercase());

    lazy_static! {
        static ref LAZY_BAZ: f64 = 12.159 * 548;
    }
}

mod external_macros {
    once_cell::external!();
    lazy_static::external!();
}

mod issue14729 {
    use once_cell::sync::Lazy;

    #[expect(clippy::non_std_lazy_statics)]
    static LAZY_FOO: Lazy<String> = Lazy::new(|| "foo".to_uppercase());
}
