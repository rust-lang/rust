fn main() {
    let _ = if true {
        42i32
    } else {
        42u32
    };
    //~^^ ERROR if and else have incompatible types
    let _ = if true { 42i32 } else { 42u32 };
    //~^ ERROR if and else have incompatible types
    let _ = if true {
        42i32;
    } else {
        42u32
    };
    //~^^ ERROR if and else have incompatible types
    let _ = if true {
        42i32
    } else {
        42u32;
    };
    //~^^ ERROR if and else have incompatible types
    let _ = if true {

    } else {
        42u32
    };
    //~^^ ERROR if and else have incompatible types
    let _ = if true {
        42i32
    } else {

    };
    //~^^^ ERROR if and else have incompatible types
}
