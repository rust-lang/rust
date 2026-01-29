//@ check-pass

trait Sendable: Send {}

fn deserialize_not_send() {
    let _: Box<dyn Sendable> = match true {
        true => loop {},
        false => (loop {}) as Box<dyn Sendable + Send>,
    };
}

fn main() {}
