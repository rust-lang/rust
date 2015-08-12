% Ergonomic error handling

Error propagation with raw `Result`s can require tedious matching and
repackaging. This tedium is largely alleviated by the `try!` macro,
and can be completely removed (in some cases) by the "`Result`-`impl`"
pattern.

### The `try!` macro

Prefer

```rust
use std::io::{File, Open, Write, IoError};

struct Info {
    name: String,
    age: i32,
    rating: i32
}

fn write_info(info: &Info) -> Result<(), IoError> {
    let mut file = File::open_mode(&Path::new("my_best_friends.txt"),
                                   Open, Write);
    // Early return on error
    try!(file.write_line(&format!("name: {}", info.name)));
    try!(file.write_line(&format!("age: {}", info.age)));
    try!(file.write_line(&format!("rating: {}", info.rating)));
    return Ok(());
}
```

over

```rust
use std::io::{File, Open, Write, IoError};

struct Info {
    name: String,
    age: i32,
    rating: i32
}

fn write_info(info: &Info) -> Result<(), IoError> {
    let mut file = File::open_mode(&Path::new("my_best_friends.txt"),
                                   Open, Write);
    // Early return on error
    match file.write_line(&format!("name: {}", info.name)) {
        Ok(_) => (),
        Err(e) => return Err(e)
    }
    match file.write_line(&format!("age: {}", info.age)) {
        Ok(_) => (),
        Err(e) => return Err(e)
    }
    return file.write_line(&format!("rating: {}", info.rating));
}
```

See
[the `result` module documentation](https://doc.rust-lang.org/stable/std/result/index.html#the-try!-macro)
for more details.

### The `Result`-`impl` pattern [FIXME]

> **[FIXME]** Document the way that the `io` module uses trait impls
> on `std::io::Result` to painlessly propagate errors.
