// Checks that lint attributes work on non-item AST nodes

fn main() {
    #[deny(unreachable_code)]
    loop {
        break;
        "unreachable"; //~ ERROR unreachable statement
    }
}
