pub fn main() {
    let x = &[1, 2, 3, 4, 5];
    if !x.is_empty() {
        let el = match x {
            [1, ..ref tail] => &tail[0],
            _ => unreachable!()
        };
        printfln!("%d", *el);
    }
}
