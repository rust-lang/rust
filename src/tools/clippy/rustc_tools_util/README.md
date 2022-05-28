# rustc_tools_util

A small tool to help you generate version information
for packages installed from a git repo

## Usage

Add a `build.rs` file to your repo and list it in `Cargo.toml`
````
build = "build.rs"
````

List rustc_tools_util as regular AND build dependency.
````
[dependencies]
rustc_tools_util = "0.1"

[build-dependencies]
rustc_tools_util = "0.1"
````

In `build.rs`, generate the data in your `main()`
````rust
fn main() {
    println!(
        "cargo:rustc-env=GIT_HASH={}",
        rustc_tools_util::get_commit_hash().unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=COMMIT_DATE={}",
        rustc_tools_util::get_commit_date().unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=RUSTC_RELEASE_CHANNEL={}",
        rustc_tools_util::get_channel().unwrap_or_default()
    );
}

````

Use the version information in your main.rs
````rust
use rustc_tools_util::*;

fn show_version() {
    let version_info = rustc_tools_util::get_version_info!();
    println!("{}", version_info);
}
````
This gives the following output in clippy:
`clippy 0.0.212 (a416c5e 2018-12-14)`


## License

Copyright 2014-2022 The Rust Project Developers

Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
<LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
option. All files in the project carrying such notice may not be
copied, modified, or distributed except according to those terms.
