// rustfmt-use_try_shorthand: true

fn main() {
    let e: Result<(), &str> = Ok(());
    let _e = e // hello comment
            .map_err(|_| "This is an error message") // Hello comment - Fails to format
        .map(|_| ());

    println!("Hello, world!");
}
