//@ check-pass

fn main() {
    let x = 1 else { return }; //~ WARN irrefutable `let...else` pattern

    // Multiline else blocks should not get printed
    let x = 1 else { //~ WARN irrefutable `let...else` pattern
        eprintln!("problem case encountered");
        return
    };
}
