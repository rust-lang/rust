#![warn(clippy::needless_character_iteration)]
#![allow(clippy::map_identity, clippy::unnecessary_operation)]

#[derive(Default)]
struct S {
    field: &'static str,
}

impl S {
    fn field(&self) -> &str {
        self.field
    }
}

fn magic(_: char) {}

fn main() {
    "foo".chars().all(|c| c.is_ascii());
    //~^ ERROR: checking if a string is ascii using iterators
    "foo".chars().any(|c| !c.is_ascii());
    //~^ ERROR: checking if a string is ascii using iterators
    "foo".chars().all(|c| char::is_ascii(&c));
    //~^ ERROR: checking if a string is ascii using iterators
    "foo".chars().any(|c| !char::is_ascii(&c));
    //~^ ERROR: checking if a string is ascii using iterators

    let s = String::new();
    s.chars().all(|c| c.is_ascii());
    //~^ ERROR: checking if a string is ascii using iterators
    "foo".to_string().chars().any(|c| !c.is_ascii());
    //~^ ERROR: checking if a string is ascii using iterators

    "foo".chars().all(|c| {
        //~^ ERROR: checking if a string is ascii using iterators
        let x = c;
        x.is_ascii()
    });
    "foo".chars().any(|c| {
        //~^ ERROR: checking if a string is ascii using iterators
        let x = c;
        !x.is_ascii()
    });

    S::default().field().chars().all(|x| x.is_ascii());
    //~^ ERROR: checking if a string is ascii using iterators

    // Should not lint!
    "foo".chars().all(|c| {
        let x = c;
        magic(x);
        x.is_ascii()
    });

    // Should not lint!
    "foo".chars().all(|c| c.is_ascii() && c.is_alphabetic());

    // Should not lint!
    "foo".chars().map(|c| c).all(|c| !char::is_ascii(&c));

    // Should not lint!
    "foo".chars().all(|c| !c.is_ascii());

    // Should not lint!
    "foo".chars().any(|c| c.is_ascii());
}
