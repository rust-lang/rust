fn main() {
    let const { 15 } = ();
    let const { foo(); bar() } = ();

    match 42 {
        const { 0 } .. const { 1 } => (),
        .. const { 0 } => (),
        const { 2 } .. => (),
    }

    let (const { () },) = ();
}
