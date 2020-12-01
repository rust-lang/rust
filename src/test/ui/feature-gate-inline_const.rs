fn main() {
    let _ = const {
        //~^ ERROR inline-const is experimental [E0658]
        true
    };
}
