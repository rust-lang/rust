//@ check-pass

pub trait Deserialize<'de>: Sized {}
pub trait DeserializeOwned: for<'de> Deserialize<'de> {}

pub trait Extensible {
    type Config;
}

// The `C` here generates a `C: Sized` candidate
pub trait Installer<C> {
    fn init<B: Extensible<Config = C>>(&mut self) -> ()
    where
        // This clause generates a `for<'de> C: Sized` candidate
        B::Config: DeserializeOwned,
    {
    }
}

fn main() {}
