macro_rules! expr {
    (no_semi) => {
        return true
    };
    (semi) => {
        return true;
    };
}

fn foo() -> bool {
    match true {
        true => expr!(no_semi),
        false if false => {
            expr!(semi)
        }
        false => {
            expr!(semi);
        }
    }
}

fn main() {}
