fn main() {
    let _ = std::panic::catch_unwind(|| Box::<str>::from("..."));
}
