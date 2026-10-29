#![warn(clippy::no_effect_replace)]

fn main() {
    let _ = "12345".replace('1', "1");
    //~^ no_effect_replace

    let _ = "12345".replace("12", "12");
    //~^ no_effect_replace

    let _ = String::new().replace("12", "12");
    //~^ no_effect_replace

    let _ = "12345".replacen('1', "1", 1);
    //~^ no_effect_replace

    let _ = "12345".replacen("12", "12", 1);
    //~^ no_effect_replace

    let _ = String::new().replacen("12", "12", 1);
    //~^ no_effect_replace

    let _ = "12345".replace("12", "22");
    let _ = "12345".replacen("12", "22", 1);

    let mut x = X::default();
    let _ = "hello".replace(&x.f(), &x.f());
    //~^ no_effect_replace

    let _ = "hello".replace(&x.f(), &x.ff());

    let _ = "hello".replace(&y(), &y());
    //~^ no_effect_replace

    let _ = "hello".replace(&y(), &z());

    let _ = Replaceme.replace("a", "a");
}

#[derive(Default)]
struct X {}

impl X {
    fn f(&mut self) -> String {
        "he".to_string()
    }

    fn ff(&mut self) -> String {
        "hh".to_string()
    }
}

fn y() -> String {
    "he".to_string()
}

fn z() -> String {
    "hh".to_string()
}

struct Replaceme;
impl Replaceme {
    pub fn replace(&mut self, a: &str, b: &str) -> Self {
        Self
    }
}
