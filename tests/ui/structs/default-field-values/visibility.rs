#![feature(default_field_values)]
pub mod foo {
    pub struct Alpha {
        beta: u8 = 42,
        gamma: bool = true,
    }
}

mod bar {
    fn baz() {
        let x = crate::foo::Alpha { .. };
        //~^ ERROR field `beta` of struct `Alpha` is private
        //~| ERROR field `gamma` of struct `Alpha` is private
    }
}

pub mod baz {
    pub struct S {
        x: i32 = 1,
    }
}
fn main() {
    let a = baz::S {
        .. //~ ERROR field `x` of struct `S` is private
    };
    let b = baz::S {
        x: 0, //~ ERROR field `x` of struct `S` is private
    };
}
