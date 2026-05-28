#![feature(default_field_values)]
pub mod foo {
    #[derive(Default)]
    pub struct Alpha {
        beta: u8 = 42,
        gamma: bool = true,
    }
}

mod bar {
    use crate::foo::Alpha;
    fn baz() {
        let _x = Alpha { .. };
        //~^ ERROR fields `beta` and `gamma` of struct `Alpha` are private
        let _x = Alpha {
            beta: 0, //~ ERROR fields `beta` and `gamma` of struct `Alpha` are private
            gamma: false,
        };
        let _x = Alpha {
            beta: 0, //~ ERROR fields `beta` and `gamma` of struct `Alpha` are private
            ..
        };
        let _x = Alpha { beta: 0, .. };
        //~^ ERROR fields `beta` and `gamma` of struct `Alpha` are private
        let _x = Alpha { beta: 0, ..Default::default() };
        //~^ ERROR fields `beta` and `gamma` of struct `Alpha` are private
    }
}

pub mod baz {
    pub struct S {
        x: i32 = 1,
    }
}
fn main() {
    let _a = baz::S {
        .. //~ ERROR field `x` of struct `S` is private
    };
    let _b = baz::S {
        x: 0, //~ ERROR field `x` of struct `S` is private
    };
}
