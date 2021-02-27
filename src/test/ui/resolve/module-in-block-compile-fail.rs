fn module_in_function_cannot_access_variables() {
    let x: i32 = 5;

    mod inner {
        use super::x;  //~ ERROR unresolved import `super::x`
        fn get_x() -> i32 {
            x
        }
    }
}

fn main() { }
