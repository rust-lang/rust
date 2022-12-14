fn f() -> f32 {
    3
    //~^ ERROR expected `;`
    5.0
}

fn k() -> f32 {
    2_u32
    //~^ ERROR expected `;`
    3_i8
    //~^ ERROR expected `;`
    5.0
}

fn main() {}
