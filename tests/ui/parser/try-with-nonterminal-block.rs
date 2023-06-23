// check-pass
// edition:2021

#![feature(try_blocks)]

macro_rules! create_try {
    ($body:block) => {
        try $body
    };
}

fn main() {
    let x: Option<&str> = create_try! {{
        None?;
        "Hello world"
    }};

    println!("{x:?}");
}
