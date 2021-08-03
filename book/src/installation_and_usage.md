# Installation and Usage

Below are instructions on how to use Clippy as a subcommand, compiled from source
or in Travis CI. Note that Clippy is installed as a 
[component](https://rust-lang.github.io/rustup/concepts/components.html?highlight=clippy#components) as part of the 
[rustup](https://rust-lang.github.io/rustup/installation/index.html) installation.

### As a cargo subcommand (`cargo clippy`)

One way to use Clippy is by installing Clippy through rustup as a cargo
subcommand.

#### Step 1: Install rustup

You can install [rustup](https://rustup.rs/) on supported platforms. This will help
us install Clippy and its dependencies.

If you already have rustup installed, update to ensure you have the latest
rustup and compiler:

```terminal
rustup update
```

#### Step 2: Install Clippy

Once you have rustup and the latest stable release (at least Rust 1.29) installed, run the following command:

```terminal
rustup component add clippy
```
If it says that it can't find the `clippy` component, please run `rustup self update`.

#### Step 3: Run Clippy

Now you can run Clippy by invoking the following command:

```terminal
cargo clippy
```

#### Automatically applying Clippy suggestions

Clippy can automatically apply some lint suggestions.
Note that this is still experimental and only supported on the nightly channel:

```terminal
cargo clippy --fix
```

#### Workspaces

All the usual workspace options should work with Clippy. For example the following command
will run Clippy on the `example` crate:

```terminal
cargo clippy -p example
```

As with `cargo check`, this includes dependencies that are members of the workspace, like path dependencies.
If you want to run Clippy **only** on the given crate, use the `--no-deps` option like this:

```terminal
cargo clippy -p example --no-deps 
```

### As a rustc replacement (`clippy-driver`)

Clippy can also be used in projects that do not use cargo. To do so, you will need to replace
your `rustc` compilation commands with `clippy-driver`. For example, if your project runs:

```terminal
rustc --edition 2018 -Cpanic=abort foo.rs
```

Then, to enable Clippy, you will need to call:

```terminal
clippy-driver --edition 2018 -Cpanic=abort foo.rs
```

Note that `rustc` will still run, i.e. it will still emit the output files it normally does.

### Continuous Integration

Adding Clippy to your continuous integration pipeline is a great way to automate the linting process. See the
[Continuous Integration](continuous_integration) chapter for more information.

