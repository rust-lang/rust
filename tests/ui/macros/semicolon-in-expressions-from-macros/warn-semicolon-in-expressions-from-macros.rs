// Ensure that trailing semicolons cause errors

macro_rules! foo {
    () => {
        true; //~ ERROR macro expansion ignores `;` and any tokens following
    }
}

fn main() {
    let _val = match true {
        true => false,
        _ => foo!()
    };
}
