//@ check-pass

fn main() {
    match true {
        true => const {}
        false => ()
    }
    const {}
    ()
}
