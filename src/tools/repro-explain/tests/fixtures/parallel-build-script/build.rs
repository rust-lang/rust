fn main() {
    let libs = vec!["ssl", "z"];
    let (tx, rx) = std::sync::mpsc::channel();

    for lib in libs {
        let tx = tx.clone();
        std::thread::spawn(move || {
            tx.send(lib.to_string()).unwrap();
        });
    }
    drop(tx);

    for lib in rx {
        println!("cargo::rustc-link-lib={lib}");
    }
}
