# Setup Guide

This guide gives a simplified opinionated setup for developers contributing to rust-analyzer using Visual Studio Code to make changes and Visual Studio Code Insiders to test those changes. This guide will assume you have Visual Studio Code and Visual Studio Code Insiders installed.

## Prerequisites

Since rust-analyzer is a Rust project, you will need to install Rust. You can download and install the latest stable version of Rust [here](https://www.rust-lang.org/tools/install).

## Step-by-Step Setup

**Step 01**: Fork the rust-analyzer repository and clone the fork to your local machine.

**Step 02**: Open the project in Visual Studio Code.

**Step 03**: Open a terminal and run `cargo build` to build the project.

**Step 04**: Install the language server locally by running the following command:

```sh
# Install only the language server  
cargo xtask install --server \  
    --code-bin code-insiders \  # Target a specific editor (code, code-exploration, code-insiders, codium, or code-oss)  
    --dev-rel  # Build in release mode with debug info level 2  
```

In the output of this command, there should be a file path provided to the installed binary on your local machine.
It should look something like the following output below:

```
Installing <path-to-rust-analyzer-binary>
Installed package `rust-analyzer v0.0.0 (<path-to-rust-analyzer-binary>)` (executable `rust-analyzer.exe`)
```

In Visual Studio Code Insiders, you will want to open your User Settings (JSON) from the Command Palette. From there you should ensure that the `rust-analyzer.server.path` key is set to the `<path-to-rust-analyzer-binary>`. This will tell Visual Studio Code Insiders to use the locally installed version that you can debug.

The User Settings (JSON) file should contain the following:

```json
{
    "rust-analyzer.server.path": "<path-to-rust-analyzer-binary>"
}
```

Now you should be able to make changes to rust-analyzer in Visual Studio Code and then view the changes in Visual Studio Code Insiders.

## Debugging rust-analyzer
The simplest way to debug rust-analyzer is to use the `eprintln!` macro. The reason why we use `eprintln!` instead of `println!` is because the language server uses `stdout` to send messages. So instead we will debug using `stderr`.

An example debugging statement could go into the `main_loop.rs` file which can be found at `crates/rust-analyzer/src/main_loop.rs`. Inside the `main_loop` we will add the following `eprintln!` to test debugging rust-analyzer:

```rs
eprintln!("Hello, world!");
```
Now, run the following commands to check the project and reinstall the server:  

```sh  
cargo check  
cargo xtask install --server --code-bin code-insiders --dev-rel  
```  

Now on Visual Studio Code Insiders, we should be able to open the Output tab on our terminal and switch to Rust Analyzer Language Server to see the `eprintln!` statement we just wrote.

If you are able to see your output, you now have a complete workflow for debugging rust-analyzer.
