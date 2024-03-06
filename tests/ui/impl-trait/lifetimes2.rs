//@ check-pass

pub fn keys<'a>(x: &'a Result<u32, u32>) -> impl std::fmt::Debug + 'a {
    match x {
        Ok(map) => Ok(map),
        Err(map) => Err(map),
    }
}

fn main() {}
