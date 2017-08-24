// rustfmt-multiline_match_arm_forces_block: false
// Option forces multiline match arm bodies to be wrapped in a block

fn main() {
    match lorem {
        Lorem::Ipsum => if ipsum {
            println!("dolor");
        },
        Lorem::Dolor => println!("amet"),
    }
}
