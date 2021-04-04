// run-pass
#![allow(non_camel_case_types)]

trait bar<T> {
    fn get_bar(&self) -> T;
}

fn foo<T, U: bar<T>>(b: U) -> T {
    b.get_bar()
}

struct cbar {
    x: isize,
}

impl bar<isize> for cbar {
    fn get_bar(&self) -> isize {
        self.x
    }
}

fn cbar(x: isize) -> cbar {
    cbar {
        x: x
    }
}

pub fn main() {
    let x: isize = foo::<isize, cbar>(cbar(5));
    assert_eq!(x, 5);
}
