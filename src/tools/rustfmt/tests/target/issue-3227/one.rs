// rustfmt-style_edition: 2015

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
