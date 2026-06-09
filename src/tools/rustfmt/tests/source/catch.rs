// rustfmt-edition: 2018
#![feature(try_blocks)]

fn main() {
    let x = try {
        foo()?
    };

    let x = try /* Invisible comment */ { foo()? };

    let x = try {
        unsafe { foo()? }
    };

    let y = match (try {
        foo()?
    }) {
        _ => (),
    };

    try {
        foo()?;
    };

    try {
        // Regular try block
    };
}
