// check-pass

use thing::*;

#[derive(Debug)]
pub enum Thing {
    Foo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thing() {
        let thing = Thing::Foo;
    }
}

mod thing {
    pub enum Thing {
        Bar,
    }
}

fn main() {}
