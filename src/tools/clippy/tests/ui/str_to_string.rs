#![warn(clippy::str_to_string)]

fn main() {
    let hello = "hello world".to_string();
    //~^ str_to_string

    let msg = &hello[..];
    msg.to_string();
    //~^ str_to_string
}

fn issue16271(key: &[u8]) {
    macro_rules! t {
        ($e:expr) => {
            match $e {
                Ok(e) => e,
                Err(e) => panic!("{} failed with {}", stringify!($e), e),
            }
        };
    }

    let _value = t!(str::from_utf8(key)).to_string();
    //~^ str_to_string
}

struct GenericWrapper<T>(T);

impl<T> GenericWrapper<T> {
    fn mapper<U, F: FnOnce(T) -> U>(self, f: F) -> U {
        f(self.0)
    }
}

fn issue16511(x: Option<&str>) {
    let _ = x.map(ToString::to_string);
    //~^ str_to_string

    let _ = x.map(str::to_string);
    //~^ str_to_string

    let _ = ["a", "b"].iter().map(ToString::to_string);
    //~^ str_to_string

    fn mapper<F: Fn(&str) -> String>(f: F) -> String {
        f("hello")
    }
    let _ = mapper(ToString::to_string);
    //~^ str_to_string

    let w = GenericWrapper("hello");
    let _ = w.mapper(ToString::to_string);
    //~^ str_to_string
}
