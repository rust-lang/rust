//@ check-pass

fn main() {
    let x = 1 else { return }; //~ WARN unreachable `else` clause

    // Multiline else blocks should not get printed
    let x = 1 else { //~ WARN unreachable `else` clause
        eprintln!("problem case encountered");
        return
    };

    let case = Some("a");
    let name = Some(case) else {
        //~^ WARN unreachable `else` clause
        //~| HELP consider using `let Some(name) = case` to match on a specific variant
        eprintln!("problem case encountered");
        return
    };
}
