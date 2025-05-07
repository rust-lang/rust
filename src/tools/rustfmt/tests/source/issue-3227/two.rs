// rustfmt-style_edition: 2024

fn main() {
    thread::spawn(|| {
        while true {
            println!("iteration");
        }
    });

    thread::spawn(|| loop {
        println!("iteration");
    });
}
