fn main() {}

fn foo() -> bool {
    while let x = 0 {
        //~^ 4:5: 7:6: mismatched types [E0308]
        return true;
    }
}
