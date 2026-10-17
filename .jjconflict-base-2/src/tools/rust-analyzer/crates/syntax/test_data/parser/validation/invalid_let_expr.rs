fn foo() {
    const _: () = let _ = None;

    let _ = if true { (let _ = None) };

    if true && (let _ = None) {
        (let _ = None);
        while let _ = None {
            match None {
                _ if let _ = None => { let _ = None; }
            }
        }
    }
}
